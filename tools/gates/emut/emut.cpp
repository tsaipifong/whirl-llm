// SPDX-License-Identifier: Apache-2.0
// arch025-emut: judge every use of a tracked class's data member as write / pointee-write / read with clang's own
// ExprMutationAnalyzer (clang/Analysis/Analyses/ExprMutationAnalyzer.h, official LLVM release, linked statically).
//
//   emut.exe --classes=Engine,JobQueue --root=R --out=F.json <tu.cpp> -- <compile flags>
//
// For every MemberExpr / DeclRefExpr that names a FieldDecl of a tracked class (a class with one of the given names,
// defined at namespace scope inside namespace whirl), the statement analyzed is the body of the innermost enclosing
// function (lambda call operators included; a constructor's member initializer / a default member initializer is its
// own statement - the way FunctionParmMutationAnalyzer treats CXXCtorInitializer), and the tool asks
//   findMutation(expr)         -> "write: <mutating stmt>"
//   findPointeeMutation(expr)  -> "pointee-write: <mutating stmt>"        (asked only when findMutation is null)
// Template instantiations are visited (MatchFinder TK_AsIs); uses inside a dependent (pattern) body are "weak".
// Calls to functions defined under --root that return a non-const reference / pointer get a call row
// (findMutation / findPointeeMutation on the call expression), so the census can propagate accessor results.
// Output (one TU): {"tu", "hits": [[file, byte_offset, member, class, "w"|"r", why, weak, returned_from_mangled]],
//                   "calls": [[file, byte_offset, callee_mangled, "w"|"r", why, returned_from_mangled, weak]],
//                   "funcs": {mangled: name}, "files": [...], "odd": [...], "nopos": n, "outside": n}
// Exit: 0 ok (clang errors are in the stderr text and in the exit code 1 of the tool run), 2 bad arguments.
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::ast_matchers;

static llvm::cl::OptionCategory Cat("emut options");
static llvm::cl::opt<std::string> OptClasses("classes", llvm::cl::desc("tracked class names, comma separated"),
                                             llvm::cl::cat(Cat));
static llvm::cl::opt<std::string> OptOut("out", llvm::cl::desc("output JSON file"), llvm::cl::cat(Cat));
static llvm::cl::opt<std::string> OptRoot("root", llvm::cl::desc("only report uses in files under this directory"),
                                          llvm::cl::cat(Cat));

namespace {

std::string normPath(llvm::StringRef p) {
  llvm::SmallString<260> s(p);
  llvm::sys::fs::make_absolute(s);
  llvm::sys::path::remove_dots(s, true);
  llvm::sys::path::native(s, llvm::sys::path::Style::windows_backslash);
  std::string r(s.str());
  for (auto& c : r) c = (char)std::tolower((unsigned char)c);
  return r;
}

struct Collector : public MatchFinder::MatchCallback {
  std::set<std::string> classes;
  std::string rootNorm;
  llvm::json::Array hits, calls, odd;
  llvm::json::Object funcs;
  std::set<std::string> files, deps;
  std::set<const void*> seen;
  std::map<const Stmt*, std::unique_ptr<ExprMutationAnalyzer>> emas;
  std::unique_ptr<ASTNameGenerator> namer;
  ASTContext* ctx = nullptr;
  int nopos = 0, outside = 0;

  bool tracked(const FieldDecl* F, std::string* cls) {
    const RecordDecl* R = F->getParent();
    if (!R || !R->getIdentifier() || !classes.count(R->getName().str())) return false;
    const DeclContext* DC = R->getDeclContext();
    if (!DC->isNamespace()) {
      if (DC->isRecord()) odd.push_back("tracked class " + R->getName().str() + " is nested (its fields are not tracked)");
      return false;
    }
    std::string q = R->getQualifiedNameAsString();
    if (q.rfind("whirl::", 0) != 0) return false;
    if (isa<ClassTemplateSpecializationDecl>(R) || cast<CXXRecordDecl>(R)->getDescribedClassTemplate()) {
      odd.push_back("tracked class " + q + " is a class template (its fields are not tracked)");
      return false;
    }
    *cls = R->getName().str();
    return true;
  }

  // innermost function body (or member initializer / default member initializer) holding N
  const Stmt* scopeOf(const DynTypedNode& start, const FunctionDecl** fn) {
    *fn = nullptr;
    DynTypedNode n = start;
    const Stmt* lastStmt = n.get<Stmt>();
    for (int guard = 0; guard < 4096; ++guard) {
      auto ps = ctx->getParents(n);
      if (ps.empty()) return lastStmt;
      const DynTypedNode& p = ps[0];
      if (const auto* L = p.get<LambdaExpr>()) {
        if (lastStmt == L->getBody()) { *fn = L->getCallOperator(); return L->getBody(); }
      }
      if (const auto* FD = p.get<FunctionDecl>()) {
        *fn = FD;
        if (FD->getBody() && lastStmt == FD->getBody()) return FD->getBody();
        return lastStmt;  // default argument or similar
      }
      if (const auto* CI = p.get<CXXCtorInitializer>()) {
        auto pp = ctx->getParents(p);
        if (!pp.empty()) *fn = pp[0].get<FunctionDecl>();
        return CI->getInit();
      }
      if (p.get<FieldDecl>() || p.get<VarDecl>() && !p.get<VarDecl>()->isLocalVarDeclOrParm()) return lastStmt;
      if (const auto* S = p.get<Stmt>()) lastStmt = S;
      n = p;
    }
    return lastStmt;
  }

  ExprMutationAnalyzer& ema(const Stmt* S) {
    auto& p = emas[S];
    if (!p) p = std::make_unique<ExprMutationAnalyzer>(*S, *ctx);
    return *p;
  }

  bool position(SourceLocation L, std::string* file, unsigned* off) {
    const SourceManager& SM = ctx->getSourceManager();
    SourceLocation E = SM.getExpansionLoc(L);
    if (E.isInvalid()) return false;
    FileID F = SM.getFileID(E);
    auto FE = SM.getFileEntryRefForID(F);
    if (!FE) return false;
    *file = normPath(FE->getName());
    *off = SM.getFileOffset(E);
    return true;
  }

  bool underRoot(const std::string& f) { return rootNorm.empty() || f.rfind(rootNorm, 0) == 0; }

  std::string describe(const Stmt* S) {
    const SourceManager& SM = ctx->getSourceManager();
    CharSourceRange R = SM.getExpansionRange(S->getSourceRange());
    std::string txt = Lexer::getSourceText(R, SM, ctx->getLangOpts()).str();
    for (auto& c : txt) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    if (txt.size() > 100) txt = txt.substr(0, 100) + "...";
    std::string loc;
    PresumedLoc P = SM.getPresumedLoc(SM.getExpansionLoc(S->getBeginLoc()));
    if (P.isValid()) loc = " @" + std::to_string(P.getLine()) + ":" + std::to_string(P.getColumn());
    return std::string(S->getStmtClassName()) + " `" + txt + "`" + loc;
  }

  std::string mangle(const FunctionDecl* F) {
    // a dependent (template pattern) function has no mangled name; the MS mangler would emit an error diagnostic
    // ("cannot mangle this template specialization type yet") and fail the TU
    if (!F || F->isTemplated() || F->isDependentContext()) return "";
    if (isa<CXXConstructorDecl>(F) || isa<CXXDestructorDecl>(F)) {
      auto v = namer->getAllManglings(F);
      return v.empty() ? "" : v[0];
    }
    return namer->getName(F);
  }

  // the mutating statement is (inside) the value of a return statement of the function: the function hands it out
  const FunctionDecl* returnedFrom(const Stmt* M, const Stmt* scope, const FunctionDecl* fn) {
    if (!fn) return nullptr;
    QualType RT = fn->getReturnType();
    if (!(RT->isReferenceType() || RT->isPointerType() || RT->isRecordType())) return nullptr;
    DynTypedNode n = DynTypedNode::create(*M);
    for (int g = 0; g < 256; ++g) {
      if (n.get<ReturnStmt>()) return fn;
      if (n.get<Stmt>() == scope) return nullptr;
      if (n.get<LambdaExpr>()) return nullptr;
      auto ps = ctx->getParents(n);
      if (ps.empty()) return nullptr;
      n = ps[0];
    }
    return nullptr;
  }

  void judge(const Expr* E, const std::string& mem, const std::string& cls) {
    if (!seen.insert(E).second) return;
    std::string file;
    unsigned off = 0;
    if (!position(E->getBeginLoc(), &file, &off)) { ++nopos; return; }
    if (!underRoot(file)) { ++outside; return; }
    files.insert(file);
    const FunctionDecl* fn = nullptr;
    const Stmt* S = scopeOf(DynTypedNode::create(*E), &fn);
    bool weak = E->isTypeDependent() || E->isValueDependent() || (fn && fn->isDependentContext());
    std::string v = "r", why = "read (findMutation and findPointeeMutation null)", rf;
    if (S) {
      if (ExprMutationAnalyzer::isUnevaluated(E, *ctx)) {
        why = "read (unevaluated)";
      } else {
        auto& A = ema(S);
        const Stmt* M = A.findMutation(E);
        const char* kind = "write: ";
        if (!M) { M = A.findPointeeMutation(E); kind = "pointee-write: "; }
        if (M) {
          v = "w";
          why = kind + describe(M);
          if (const FunctionDecl* R = returnedFrom(M, S, fn)) { rf = mangle(R); why += "; returned by " + R->getNameAsString(); }
        }
      }
    } else {
      why = "read (no enclosing statement)";
    }
    if (weak) why += " (weak)";
    hits.push_back(llvm::json::Array{file, off, mem, cls, v, why, weak, rf.empty() ? llvm::json::Value(nullptr) : llvm::json::Value(rf)});
  }

  void judgeCall(const CallExpr* C, const FunctionDecl* callee) {
    if (!seen.insert(C).second) return;
    std::string file, dfile;
    unsigned off = 0, doff = 0;
    const FunctionDecl* def = nullptr;
    if (!callee->hasBody(def) || !def) return;
    if (!position(def->getLocation(), &dfile, &doff) || !underRoot(dfile)) return;
    if (!position(C->getBeginLoc(), &file, &off) || !underRoot(file)) return;
    QualType RT = callee->getReturnType();
    bool ref = RT->isReferenceType() && !RT.getNonReferenceType().isConstQualified();
    bool ptr = RT->isPointerType() && !RT->getPointeeType().isConstQualified();
    if (!ref && !ptr) return;
    const FunctionDecl* fn = nullptr;
    const Stmt* S = scopeOf(DynTypedNode::create(*C), &fn);
    bool weak = C->isTypeDependent() || (fn && fn->isDependentContext());
    std::string v = "r", why = ref ? "reference result read" : "pointer result: pointee read", rf;
    if (S && !ExprMutationAnalyzer::isUnevaluated(C, *ctx)) {
      auto& A = ema(S);
      const Stmt* M = ref ? A.findMutation(C) : A.findPointeeMutation(C);
      if (ref && !M) M = A.findPointeeMutation(C);
      if (M) {
        v = "w";
        why = std::string(ref ? "write: " : "pointee-write: ") + describe(M);
        if (const FunctionDecl* R = returnedFrom(M, S, fn)) { rf = mangle(R); why += "; returned by " + R->getNameAsString(); }
      }
    }
    funcs[mangle(callee)] = callee->getNameAsString();
    calls.push_back(llvm::json::Array{file, off, mangle(callee), v, why, rf.empty() ? llvm::json::Value(nullptr) : llvm::json::Value(rf), weak});
  }

  // every user file the TU read (the census's "deps": which files' bodies this TU covers)
  void onEndOfTranslationUnit() override {
    if (!ctx) return;
    const SourceManager& SM = ctx->getSourceManager();
    for (auto it = SM.fileinfo_begin(); it != SM.fileinfo_end(); ++it) {
      std::string f = normPath(it->getFirst().getName());
      if (underRoot(f)) deps.insert(f);
    }
  }

  void run(const MatchFinder::MatchResult& R) override {
    if (!ctx) { ctx = R.Context; namer = std::make_unique<ASTNameGenerator>(*ctx); }
    std::string cls;
    if (const auto* M = R.Nodes.getNodeAs<MemberExpr>("me")) {
      if (const auto* F = dyn_cast<FieldDecl>(M->getMemberDecl()))
        if (tracked(F, &cls)) judge(M, F->getName().str(), cls);
    } else if (const auto* D = R.Nodes.getNodeAs<DeclRefExpr>("dre")) {
      if (const auto* F = dyn_cast<FieldDecl>(D->getDecl()))
        if (tracked(F, &cls)) judge(D, F->getName().str(), cls);
    } else if (const auto* C = R.Nodes.getNodeAs<CallExpr>("call")) {
      if (const FunctionDecl* F = C->getDirectCallee()) judgeCall(C, F);
    } else if (const auto* F = R.Nodes.getNodeAs<FunctionDecl>("fn")) {
      std::string f;
      unsigned o;
      if (F->doesThisDeclarationHaveABody() && position(F->getLocation(), &f, &o) && underRoot(f)) {
        std::string m = mangle(F);
        if (!m.empty()) funcs[m] = F->getNameAsString();
      }
    }
  }
};

}  // namespace

int main(int argc, const char** argv) {
  auto P = tooling::CommonOptionsParser::create(argc, argv, Cat);
  if (!P) { llvm::errs() << P.takeError(); return 2; }
  if (OptOut.empty() || OptClasses.empty()) { llvm::errs() << "emut: --out and --classes are required\n"; return 2; }
  Collector C;
  llvm::StringRef cs(OptClasses);
  llvm::SmallVector<llvm::StringRef, 32> parts;
  cs.split(parts, ',', -1, false);
  for (auto p : parts) C.classes.insert(p.trim().str());
  if (!OptRoot.empty()) { C.rootNorm = normPath(OptRoot); if (C.rootNorm.back() != '\\') C.rootNorm += '\\'; }
  MatchFinder F;
  F.addMatcher(traverse(TK_AsIs, memberExpr(member(fieldDecl())).bind("me")), &C);
  F.addMatcher(traverse(TK_AsIs, declRefExpr(to(fieldDecl())).bind("dre")), &C);
  F.addMatcher(traverse(TK_AsIs, callExpr(callee(functionDecl())).bind("call")), &C);
  F.addMatcher(traverse(TK_AsIs, functionDecl(isDefinition()).bind("fn")), &C);
  tooling::ClangTool T(P->getCompilations(), P->getSourcePathList());
  int rc = T.run(tooling::newFrontendActionFactory(&F).get());
  llvm::json::Array files, deps;
  for (auto& f : C.files) files.push_back(f);
  for (auto& f : C.deps) deps.push_back(f);
  llvm::json::Object out{{"tu", P->getSourcePathList().empty() ? "" : P->getSourcePathList()[0]},
                         {"rc", rc},
                         {"hits", std::move(C.hits)},
                         {"calls", std::move(C.calls)},
                         {"funcs", std::move(C.funcs)},
                         {"files", std::move(files)},
                         {"deps", std::move(deps)},
                         {"odd", std::move(C.odd)},
                         {"nopos", C.nopos},
                         {"outside", C.outside},
                         {"analyzer", "clang ExprMutationAnalyzer (LLVM " LLVM_VERSION_STRING ")"}};
  std::error_code ec;
  llvm::raw_fd_ostream os(OptOut, ec);
  if (ec) { llvm::errs() << "emut: cannot write " << OptOut << ": " << ec.message() << "\n"; return 2; }
  os << llvm::json::Value(std::move(out)) << "\n";
  return rc;
}

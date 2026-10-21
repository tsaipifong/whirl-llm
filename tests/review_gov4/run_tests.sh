#!/bin/bash
# REVIEW-GOV-4: export include/ + src/common/ of fbdbe38 and cb6c9b4, build and run the two independent tests on each
set -u
W=/c/ClaudeWorkspace/whirl-work
S=$W/scratch/gov4review
T=$W/wt_review_gov2/tests/review_gov4
TW=$(cygpath -w $T)
for c in fbdbe38 cb6c9b4; do
  rm -rf $S/tree_$c; mkdir -p $S/tree_$c
  git -C $W/wt_governor archive $c include src/common | tar -x -C $S/tree_$c
done
for c in fbdbe38 cb6c9b4; do
  TR=$(cygpath -w $S/tree_$c)
  fb=""; [ $c = fbdbe38 ] && fb=FB
  echo "######## C-2 test on $c ($(git -C $W/wt_governor rev-parse --short $c))"
  cmd //c "$TW\build_c2.bat" "$TR" "$(cygpath -w $S/out_c2_$c)" >/dev/null && $S/out_c2_$c/c2_test.exe; echo "exit $?"
  echo "######## pool test on $c"
  cmd //c "$TW\build_pool.bat" "$TR" "$(cygpath -w $S/out_pool_$c)" $fb >/dev/null && $S/out_pool_$c/pool_test.exe; echo "exit $?"
done

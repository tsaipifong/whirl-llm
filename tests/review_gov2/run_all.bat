@echo off
rem REVIEW-GOV-2: rebuild + run the reviewer's tests in b_old (3a4febd) and b_new (14ba714)
set S=C:\ClaudeWorkspace\whirl-work\scratch\gov2review
set T=C:\ClaudeWorkspace\whirl-work\wt_review_gov2\tests\review_gov2
set G=C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf
call %T%\rebuild.bat %S%\b_old > %S%\rb2_b_old.txt 2>&1
echo old build rc=%ERRORLEVEL% > %S%\rb2_rc.txt
call %T%\rebuild.bat %S%\b_new > %S%\rb2_b_new.txt 2>&1
echo new build rc=%ERRORLEVEL% >> %S%\rb2_rc.txt
%S%\b_old\review-gov2-restore.exe --gguf %G% > %S%\restore2_old.txt 2>&1
echo exit=%ERRORLEVEL% >> %S%\restore2_old.txt
%S%\b_new\review-gov2-restore.exe --gguf %G% > %S%\restore2_new.txt 2>&1
echo exit=%ERRORLEVEL% >> %S%\restore2_new.txt
%S%\b_old\review-gov2-unit.exe > %S%\unit2_old.txt 2>&1
echo exit=%ERRORLEVEL% >> %S%\unit2_old.txt
%S%\b_new\review-gov2-unit.exe > %S%\unit2_new.txt 2>&1
echo exit=%ERRORLEVEL% >> %S%\unit2_new.txt

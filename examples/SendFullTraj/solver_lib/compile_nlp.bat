@echo off
set source_file=%1
set target_file=%2

set "lockdir=D:\01_Git\02_Python\AgentLib-FlexQuant_Ecom\AgentLib-FlexQuant\examples\SendFullTraj\solver_lib\code_gen\vcvarsall_lock"

:waitlock
mkdir "%lockdir%" 2>nul
if errorlevel 1 (
    ping -n 2 127.0.0.1 >nul
    goto waitlock
)

:: We have the lock, setup environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Release lock
rmdir "%lockdir%"

:: Compile
cl /LD /FS /Z7 %source_file%
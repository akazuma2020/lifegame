@echo off
setlocal EnableExtensions

rem Build bridge.dll with the MSYS2 MinGW toolchain.
rem Run this from an MSYS2 UCRT64/MINGW64 shell, or from a Windows prompt
rem where the selected MSYS2 bin directory is available.

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" || exit /b 2
if not defined MSYS2_ROOT set "MSYS2_ROOT=C:\msys64"

rem Prefer one MSYS2 prefix containing both wgpu-native and SDL2.
if exist "%MSYS2_ROOT%\ucrt64\bin\gcc.exe" if exist "%MSYS2_ROOT%\ucrt64\bin\wgpu_native.dll" if exist "%MSYS2_ROOT%\ucrt64\bin\SDL2.dll" (
  set "MSYS2_PREFIX=%MSYS2_ROOT%\ucrt64"
  goto :toolchain_ready
)
if exist "%MSYS2_ROOT%\mingw64\bin\gcc.exe" if exist "%MSYS2_ROOT%\mingw64\bin\wgpu_native.dll" if exist "%MSYS2_ROOT%\mingw64\bin\SDL2.dll" (
  set "MSYS2_PREFIX=%MSYS2_ROOT%\mingw64"
  goto :toolchain_ready
)

echo [ERROR] A complete MSYS2 MinGW prefix was not found.
echo Install from an UCRT64 shell with:
echo   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-wgpu-native mingw-w64-ucrt-x86_64-SDL2
goto :config_fail

:toolchain_ready
set "MSYS2_BIN=%MSYS2_PREFIX%\bin"
set "CC=%MSYS2_BIN%\gcc.exe"
set "WGPU_INCLUDE_DIR=%MSYS2_PREFIX%\include"
set "SDL2_INCLUDE_DIR=%MSYS2_PREFIX%\include\SDL2"
set "LIB_DIR=%MSYS2_PREFIX%\lib"
set "OUTPUT_DLL=%SCRIPT_DIR%bridge.dll"

if not exist "%WGPU_INCLUDE_DIR%\webgpu\wgpu.h" (
  echo [ERROR] wgpu-native headers were not found under %WGPU_INCLUDE_DIR%.
  goto :config_fail
)
if not exist "%SDL2_INCLUDE_DIR%\SDL.h" (
  echo [ERROR] SDL2 headers were not found under %SDL2_INCLUDE_DIR%.
  goto :config_fail
)

echo Using MSYS2 MinGW toolchain: %CC%
echo Using MSYS2 wgpu-native and SDL2 from: %MSYS2_PREFIX%

"%CC%" -shared -O2 -std=c11 -Wall -Wextra -Werror -static-libgcc ^
  -DWIN32_LEAN_AND_MEAN -DNOMINMAX -DWGPU_SHARED_LIBRARY ^
  -I"%WGPU_INCLUDE_DIR%" -I"%SDL2_INCLUDE_DIR%" ^
  bridge.c -L"%LIB_DIR%" -lwgpu_native -lSDL2 ^
  -o "%OUTPUT_DLL%"
if errorlevel 1 goto :fail

echo Built %OUTPUT_DLL%
echo Runtime DLLs remain in %MSYS2_BIN%; add this directory to PATH before starting SBCL.

if /I "%~1"=="validate" (
  "%CC%" -O2 -std=c11 -Wall -Wextra -Werror -static-libgcc ^
    -DWIN32_LEAN_AND_MEAN -DNOMINMAX -DWGPU_SHARED_LIBRARY -DLIFE_SHADER_VALIDATE ^
    -I"%WGPU_INCLUDE_DIR%" -I"%SDL2_INCLUDE_DIR%" ^
    bridge.c -L"%LIB_DIR%" -lwgpu_native -lSDL2 ^
    -o validate-shader.exe
  if errorlevel 1 goto :fail
  validate-shader.exe ..\shader.wgsl
  if errorlevel 1 goto :fail
  del /q validate-shader.exe
)

popd
exit /b 0

:config_fail
popd
exit /b 2

:fail
echo [ERROR] MinGW build failed.
popd
exit /b 1

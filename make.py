from typing import List
import subprocess
import sys, os

RAYLIB_LIBS = [
    "-I.\\src\\raylib-windows\\include", "-lraylib", "-L.\\src\\raylib-windows\\lib",
    "-lOpenGL32", "-lmsvcrt", "-lGdi32", "-lWinMM", "-lkernel32", "-lshell32", "-lUser32",
    "-Xlinker", "/NODEFAULTLIB:LIBCMT",
] if os.name == "nt" else [
    "-I./src/raylib-linux/include/", "-l:libraylib.a", "-L./src/raylib-linux/lib/",
]

PROJECTS = ["moving", "mandelbrot", "gol", "mandelbulb", "balls"]

def call_cmd(cmd: List[str]) -> int:
    print("[CMD] " + " ".join(cmd))
    return subprocess.call(cmd)

def call_and_catch(cmd: List[str], err_msg: str):
    code = call_cmd(cmd)
    if code != 0:
        print("[ERROR] Command exited with code", code)
        print(err_msg)
        exit(code)

def print_usage_and_exit(prog: str):
    print(f"Usage: {prog} [mode]")
    print("Supported modes:")
    print(" - run     <project>   Compile and run a given project, for example `make.py run mandel`.")
    print(" - compile <project>   Compile a given project, for example `make.py compile mandel`.")
    exit(1)

def compile(prog: str, proj: str, nvcc_args: str, raylib: bool = True):
    _args = [] if len(nvcc_args) == 0 else nvcc_args.split(" ")
    cmd = [
        "nvcc", f"./src/{proj}.cu",
        "-std=c++17",
        "-O3",
    ]
    if len(_args) != 0:
        cmd.extend(_args)
    cmd.extend(["-o", f"./build/{proj}"])
    if not os.path.exists('./build'):
        os.makedirs('./build')
    if proj not in PROJECTS:
        print(f"error: Can't compile unknown project `{proj}`.")
        print_usage_and_exit(prog)
    if raylib:
        cmd.extend(RAYLIB_LIBS)
    call_and_catch(cmd, f"Could not compile `{proj}`")

def run(prog: str, proj: str, nvcc_args: str):
    compile(prog, proj, nvcc_args)
    if os.name == "posix":
        call_and_catch([f"./build/{proj}"], f"Could not run `{proj}`.")
    else:
        call_and_catch([f".\\build\\{proj}"], f"Could not run `{proj}`.")

def main():
    prog = sys.argv[0]
    if len(sys.argv) <= 1:
        print("error: no mode specified")
        print_usage_and_exit(prog)
    mode = sys.argv[1]
    args = sys.argv[3] if len(sys.argv) >= 4 else ""
    if mode == "run":
        if len(sys.argv) <= 2:
            print(f"error: `{mode}` expects a project to run.")
            print_usage_and_exit(prog)
        proj = sys.argv[2]
        run(prog, proj, args)
    elif mode == "compile":
        if len(sys.argv) <= 2:
            print(f"error: `{mode}` expects a project to compile.")
            print_usage_and_exit(prog)
        proj = sys.argv[2]
        compile(prog, proj, args)
    else:
        print(f"error: unknown mode `{mode}`.")
        print_usage_and_exit(prog)

if __name__ == "__main__":
    main()

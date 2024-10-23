from typing import List
import subprocess
import sys

def call_cmd(cmd: List[str]) -> int:
    print("[CMD] " + " ".join(cmd))
    return subprocess.call(cmd)

def call_and_catch(cmd: List[str], err_msg: str):
    code = call_cmd(cmd)
    if code != 0:
        print(err_msg)
        exit(code)

def print_usage_and_exit(prog: str):
    print(f"Usage: {prog} [mode]")
    print("Supported modes:")
    print(" - run     <project>   Compile and run a given project, for example `make.py run mandel`.")
    print(" - compile <project>   Compile a given project, for example `make.py compile mandel`.")
    exit(1)

def compile(prog: str, proj: str):
    match proj:
        case "moving":
            call_and_catch([
                "nvcc", f"./src/{proj}.cu",
                "-std=c++17",
                "-O3",
                "-o", f"./build/{proj}",
            ], f"Could not compile `{proj}`")
        case "mandel":
            call_and_catch([
                "nvcc", f"./src/{proj}.cu",
                "-std=c++17",
                "-I.\\src\\raylib\\include", "-lraylib", "-L.\\src\\raylib\\lib",
                "-lOpenGL32", "-lmsvcrt", "-lGdi32", "-lWinMM", "-lkernel32", "-lshell32", "-lUser32",
                "-Xlinker", "/NODEFAULTLIB:LIBCMT",
                "-O3",
                "-o", f"./build/{proj}",
            ], f"Could not compile `{proj}`")
        case other:
            print(f"error: Can't compile unknown project `{other}`.")
            print_usage_and_exit(prog)

def run(prog: str, proj: str):
    compile(prog, proj)
    call_and_catch([f".\\build\\{proj}"], f"Could not run `{proj}`.")

def main():
    prog = sys.argv[0]
    if len(sys.argv) <= 1:
        print("error: no mode specified")
        print_usage_and_exit(prog)
    mode = sys.argv[1]
    if mode == "run":
        if len(sys.argv) <= 2:
            print(f"error: `{mode}` expects a project to run.")
            print_usage_and_exit(prog)
        proj = sys.argv[2]
        run(prog, proj)
    elif mode == "compile":
        if len(sys.argv) <= 2:
            print(f"error: `{mode}` expects a project to compile.")
            print_usage_and_exit(prog)
        proj = sys.argv[2]
        compile(prog, proj)
    else:
        print(f"error: unknown mode `{mode}`.")
        print_usage_and_exit(prog)

if __name__ == "__main__":
    main()

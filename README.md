# SchedulingWithGPU

## Results and Report
- https://github.com/GabrieleMessina/Parallel-implementation-on-GPU-of-the-PETS-task-scheduling-algorithm

## Prerequisites
- `C` and `C++` compilers.
- OpenCL SDK for at least one of your platforms.
- An editor, of course.

## Build (Windows)
1. Check the `makefile` to suits your environment variables and compiler.
2. Open _Visual Studio Developer Command Prompt_, move to the `src` folder and send the `make` command.
4. Run the desired `.exe` file.
5. Eventually choose the platform you want the code to run by writing his index in the `config.txt` file, you can find the correct index using `oclinfo.exe` once you've build it.

## Build (Linux [and i think Mac too])
1. Same as Windows but you don't need to build from _Visual Studio Developer Command Prompt_, every shell will be alright.
2. And of course you won't have any `.exe` file, just run whatever the compiler will create.

## Required packages
```bash
sudo apt install cmake
sudo apt install libsfml-dev
```

## Installation
Use the following command pattern to launch the installation
```bash
sh install.sh target build_mode
```

__target__ must be replaced with one of the following values: [dev/shared/static/example]
* __target = dev__ by default if not defined

__build_mode__ must be replaced as well with : [debug/release]
* __build_mode = debug__ by default if not defined

For instance, if I choose to build a shared version of the library in release mode:
```bash
sh install.sh shared release
```

If I choose to build the sources files in debug mode to get an executable for developpement purpose :
```bash
sh install.sh dev debug
# sh install.sh debug - Valid
# sh install.sh - Valid
```

In the case where __target=example__, build the static library in release mode before using this target because the example project (in 'example' directory) use it
```bash
sh install.sh static release
sh install.sh example
```

Then the binairies will appear in the "bin" directory

## Documentation

### Create

## Running Tests

All unit tests can be run with `python -m unittest discover -s tests -v`

## Running headless instance of Starcraft II for replay simulation

### MacOS

On a MacOS device that has the retail version of Starcraft II installed in the default applications folder, use  
`"/Applications/StarCraft II/Support/SC2Switcher.app/Contents/MacOS/SC2Switcher" -listen 127.0.0.1 -port 5000 -displayMode 0 -launch`  
to launch Starcraft II in headless mode on port 5000 and listen for client requests

### Windows

On a Windows machine, use

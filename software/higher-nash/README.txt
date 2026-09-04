# Fixed Higher Nash interactive app

Upload these three files together in the same GitHub Pages directory:

- index.html
- Funciones.py
- web_engine.py

Example:
software/higher-nash/

Then link to:
software/higher-nash/index.html

The previous version had two relevant problems:
1. NetworkX was installed through micropip even though it is included in the Pyodide distribution.
2. The web adapter created a second set of SymPy symbols instead of using the x,y,z,t symbols from Funciones.py. That breaks symbolic differentiation/substitution.

This version fixes both. It also reports the actual initialization error on the page.

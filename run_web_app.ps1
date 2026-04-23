$root = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$root\.venv\Scripts\python.exe" "$root\web_app.py"

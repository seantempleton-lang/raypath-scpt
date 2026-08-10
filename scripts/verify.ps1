$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "The RayPath SCPT regression suite failed."
    }

    python raypath_scpt.py --self-test
    if ($LASTEXITCODE -ne 0) {
        throw "The RayPath SCPT numerical self-test failed."
    }

    python -m compileall -q raypath_core raypath_reporting.py raypath_scpt.py tests
    if ($LASTEXITCODE -ne 0) {
        throw "Python compilation checks failed."
    }
}
finally {
    Pop-Location
}

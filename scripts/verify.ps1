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

    python -m py_compile raypath_scpt.py tests\test_core.py tests\test_project_state.py
    if ($LASTEXITCODE -ne 0) {
        throw "Python compilation checks failed."
    }
}
finally {
    Pop-Location
}

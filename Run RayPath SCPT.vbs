Option Explicit

Dim shell, fileSystem, candidates, baseDirectory, scriptPath
Dim configuredPython, pathEntry, candidate, selectedPython, selectedPythonw, pythonwCandidate
Dim arguments, argument, command, exitCode, waitForExit

Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")
Set candidates = CreateObject("Scripting.Dictionary")

baseDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
scriptPath = fileSystem.BuildPath(baseDirectory, "raypath_scpt.py")

Sub AddCandidate(value)
    Dim resolved
    If Len(Trim(value)) = 0 Then Exit Sub
    resolved = shell.ExpandEnvironmentStrings(value)
    If fileSystem.FileExists(resolved) Then
        If Not candidates.Exists(LCase(resolved)) Then candidates.Add LCase(resolved), resolved
    End If
End Sub

configuredPython = shell.Environment("PROCESS")("RAYPATH_SCPT_PYTHON")
AddCandidate configuredPython

For Each pathEntry In Split(shell.ExpandEnvironmentStrings("%PATH%"), ";")
    If Len(Trim(pathEntry)) > 0 Then AddCandidate fileSystem.BuildPath(pathEntry, "python.exe")
Next

AddCandidate "C:\ProgramData\anaconda3\python.exe"

selectedPython = ""
selectedPythonw = ""
For Each candidate In candidates.Items
    command = Quote(candidate) & " -c " & Quote("import numpy, scipy, matplotlib, PySide6, reportlab")
    exitCode = shell.Run(command, 0, True)
    If exitCode = 0 Then
        pythonwCandidate = fileSystem.BuildPath(fileSystem.GetParentFolderName(candidate), "pythonw.exe")
        If fileSystem.FileExists(pythonwCandidate) Then
            selectedPython = candidate
            selectedPythonw = pythonwCandidate
            Exit For
        End If
    End If
Next

If Len(selectedPythonw) = 0 Then
    MsgBox "RayPath SCPT could not find a compatible Python installation with pythonw.exe and all required dependencies." & vbCrLf & vbCrLf & _
           "Run the diagnostic .cmd launcher for more information.", vbCritical, "RayPath SCPT"
    WScript.Quit 1
End If

arguments = ""
For Each argument In WScript.Arguments
    arguments = arguments & " " & Quote(CStr(argument))
Next

command = Quote(selectedPythonw) & " " & Quote(scriptPath) & arguments
waitForExit = (WScript.Arguments.Count > 0)
exitCode = shell.Run(command, 1, waitForExit)

If waitForExit Then WScript.Quit exitCode

Function Quote(value)
    Quote = Chr(34) & Replace(CStr(value), Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function

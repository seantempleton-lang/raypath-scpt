Option Explicit

Dim shell, fileSystem, candidates, baseDirectory, scriptPath
Dim configuredPython, pathEntry, candidate, selectedPythonw
Dim pythonwCandidate, arguments, argument, command, exitCode, waitForExit

Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")
Set candidates = CreateObject("Scripting.Dictionary")

baseDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
scriptPath = fileSystem.BuildPath(baseDirectory, "raypath_scpt.py")
shell.CurrentDirectory = baseDirectory

If Not fileSystem.FileExists(scriptPath) Then
    MsgBox "RayPath SCPT could not find its application entry point:" & vbCrLf & vbCrLf & _
           scriptPath, vbCritical, "RayPath SCPT"
    WScript.Quit 1
End If

Sub AddCandidate(value)
    Dim resolved
    If IsNull(value) Or IsEmpty(value) Then Exit Sub
    If Len(Trim(CStr(value))) = 0 Then Exit Sub

    resolved = shell.ExpandEnvironmentStrings(CStr(value))
    If fileSystem.FileExists(resolved) Then
        If Not candidates.Exists(LCase(resolved)) Then
            candidates.Add LCase(resolved), resolved
        End If
    End If
End Sub

Sub AddPythonLauncherCandidates()
    Dim process, line, driveMarker

    On Error Resume Next
    Set process = shell.Exec("py -0p")
    If Err.Number <> 0 Then
        Err.Clear
        On Error GoTo 0
        Exit Sub
    End If
    On Error GoTo 0

    Do Until process.StdOut.AtEndOfStream
        line = Trim(process.StdOut.ReadLine())
        driveMarker = InStr(line, ":\")
        If driveMarker > 1 Then AddCandidate Mid(line, driveMarker - 1)
    Loop
End Sub

' Prefer the project's environment, followed by an explicitly configured
' interpreter and the normal Windows Python discovery locations.
AddCandidate fileSystem.BuildPath(baseDirectory, ".venv\Scripts\python.exe")
configuredPython = shell.Environment("PROCESS")("RAYPATH_SCPT_PYTHON")
AddCandidate configuredPython
AddPythonLauncherCandidates

For Each pathEntry In Split(shell.ExpandEnvironmentStrings("%PATH%"), ";")
    If Len(Trim(pathEntry)) > 0 Then
        AddCandidate fileSystem.BuildPath(pathEntry, "python.exe")
    End If
Next

AddCandidate "C:\ProgramData\anaconda3\python.exe"

selectedPythonw = ""
For Each candidate In candidates.Items
    command = Quote(candidate) & " -c " & _
              Quote("import numpy, scipy, matplotlib, PySide6.QtCore, reportlab, raypath_core")
    exitCode = shell.Run(command, 0, True)
    If exitCode = 0 Then
        pythonwCandidate = fileSystem.BuildPath( _
            fileSystem.GetParentFolderName(candidate), "pythonw.exe")
        If fileSystem.FileExists(pythonwCandidate) Then
            selectedPythonw = pythonwCandidate
            Exit For
        End If
    End If
Next

If Len(selectedPythonw) = 0 Then
    MsgBox "RayPath SCPT could not find a compatible Python installation with " & _
           "pythonw.exe and the required dependencies." & vbCrLf & vbCrLf & _
           "Set RAYPATH_SCPT_PYTHON to a compatible python.exe, or install the " & _
           "packages in requirements.txt. Run scripts\verify.cmd for diagnostics.", _
           vbCritical, "RayPath SCPT"
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

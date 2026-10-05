Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = folder
cmd = "cmd /c python -m pip install -r """ & folder & "\requirements.txt"" && python -m streamlit run """ & folder & "\app.py"""
sh.Run cmd, 0, False
WScript.Sleep 5000
sh.Run "http://localhost:8501", 1, False

### 在windows下使用
```ps
$Env:PATH='D:\workspace\cpp_projects\local_ai_runtime\build-vs2022-x64\bin\Release;' + $Env:PATH; $Env:RUNTIME_LISTEN_HOST='127.0.0.1'; $Env:RUNTIME_LISTEN_PORT='18080'; $Env:RUNTIME_PROVIDER='lmdeploy'; $Env:LMDEPLOY_HOST='http://127.0.0.1:23333'; $Env:NO_PROXY='127.0.0.1,localhost'; $Env:no_proxy='127.0.0.1,localhost'; & 'D:\workspace\cpp_projects\local_ai_runtime\build-vs2022-x64\Release\local-ai-runtime.exe'
```
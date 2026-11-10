#!/usr/bin/env bash
# usage: put.sh localfile remote_relpath
curl -s --cert ~/.ssl/cadcproxy.pem -X PUT -T "$1" -o /dev/null -w "$2 %{http_code}\n" https://ws-uv.canfar.net/arc/files/projects/ots/adass2026/$2

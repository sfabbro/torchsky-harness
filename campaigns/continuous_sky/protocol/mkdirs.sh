#!/usr/bin/env bash
# usage: mkdirs.sh path1 path2 ... (relative to /projects/ots/adass2026)
for p in "$@"; do
curl -s --cert ~/.ssl/cadcproxy.pem -X PUT -H 'Content-Type: application/xml' -o /dev/null -w "$p %{http_code}\n" \
 --data "<vos:node xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:vos=\"http://www.ivoa.net/xml/VOSpace/v2.0\" uri=\"vos://cadc.nrc.ca~arc/projects/ots/adass2026/$p\" xsi:type=\"vos:ContainerNode\"><vos:properties/><vos:nodes/></vos:node>" \
 https://ws-uv.canfar.net/arc/nodes/projects/ots/adass2026/$p
done

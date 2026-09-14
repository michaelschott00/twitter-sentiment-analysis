# Runpod MCP server (https://github.com/runpod/runpod-mcp) exposed over
# Streamable HTTP so other compose services (e.g. opencode running in `dev`)
# can reach it at http://runpod-mcp:8001/mcp.
#
# The published npm package only ships stdio/http library entrypoints, so this
# image installs @runpod/mcp-server and adds a tiny wrapper (server.mjs) that
# listens on 0.0.0.0 and forwards each request to the upstream HTTP handler.
# When the client sends no Authorization header, the wrapper injects
# RUNPOD_API_KEY (provided via `env_file` in compose.yaml) as the Bearer token.

FROM node:22-alpine

WORKDIR /app

# Pinned version (single source of truth for reproducibility); bump to upgrade.
ARG RUNPOD_MCP_VERSION=3.4.0

RUN npm install "@runpod/mcp-server@${RUNPOD_MCP_VERSION}"

# Minimal HTTP wrapper around the upstream `./http` export.
COPY <<EOF ./server.mjs
import { createServer } from 'node:http';
import { handleMcpRequest } from '@runpod/mcp-server/http';

const PORT = Number(process.env.PORT ?? '8001');
const API_KEY = process.env.RUNPOD_API_KEY;

if (!API_KEY) {
  console.error('warning: RUNPOD_API_KEY is not set; requests without an Authorization header will be rejected');
}

const server = createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('ok');
    return;
  }
  // The upstream handler requires a per-request Bearer token (Runpod API key
  // or OAuth token) and holds no credential itself. Inject the server-side key
  // when the client did not supply one, so plain remote clients just work.
  if (API_KEY && !req.headers.authorization) {
    req.headers.authorization = 'Bearer ' + API_KEY;
  }
  handleMcpRequest(req, res, { serverVersion: 'docker' }).catch((err) => {
    console.error('mcp handler error:', err);
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'internal error' }));
    }
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.error('runpod-mcp listening on http://0.0.0.0:' + PORT + '/mcp');
});
EOF

ENV PORT=8001
EXPOSE 8001

CMD ["node", "server.mjs"]

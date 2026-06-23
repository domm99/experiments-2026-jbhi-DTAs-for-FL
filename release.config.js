var prepareCmd = `
echo VERSION="\${nextRelease.version}" > .env
echo PROJECT_NAME= 'Experiments for JBHI journal -- DTAs for FL in Smart Healthcare' >> .env
docker build -t davidedomini99/experiments-2026-jbhi-dtas-for-fl:\${nextRelease.version} .
`
var publishCmd = `
docker push davidedomini99/experiments-2026-jbhi-dtas-for-fl:\${nextRelease.version}
git add .env
git commit -m "chore(release): update .env versions to \${nextRelease.version} [skip ci]"
git push
`
var config = require('semantic-release-preconfigured-conventional-commits');
config.plugins.push(
    ["@semantic-release/exec", {
        "prepareCmd": prepareCmd,
        "publishCmd": publishCmd,
    }],
    ["@semantic-release/github", {
        "assets": [
            { "path": "charts.tar.zst" },
        ]
    }],
    "@semantic-release/git",
)
module.exports = config
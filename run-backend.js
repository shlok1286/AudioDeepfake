const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const rootDir = __dirname;
let pythonCmd = 'python';

const winVenv = path.join(rootDir, '.venv', 'Scripts', 'python.exe');
const unixVenv = path.join(rootDir, '.venv', 'bin', 'python');

if (fs.existsSync(winVenv)) {
  pythonCmd = winVenv;
} else if (fs.existsSync(unixVenv)) {
  pythonCmd = unixVenv;
}

console.log(`[VerifyVoice] Starting Python ML backend using: ${pythonCmd}`);
const mainPy = path.join(rootDir, 'backend', 'main.py');
const child = spawn(pythonCmd, [mainPy], {
  cwd: rootDir,
  stdio: 'inherit',
  env: { ...process.env, PYTHONUNBUFFERED: '1' }
});

child.on('error', (err) => {
  console.error('[VerifyVoice] Error launching backend process:', err);
  process.exit(1);
});

child.on('exit', (code) => {
  process.exit(code ?? 0);
});

process.on('SIGINT', () => {
  child.kill('SIGINT');
});
process.on('SIGTERM', () => {
  child.kill('SIGTERM');
});

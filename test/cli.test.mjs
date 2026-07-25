/**
 * CLI parsing regression tests.
 */

import { spawn } from 'node:child_process';
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = dirname(__dirname);
const cliArgsEnvName = 'SPLAT_TRANSFORM_CLI_TEST_ARGS';

const cliBootstrap = `
const cliArgs = JSON.parse(process.env.${cliArgsEnvName});
process.argv = ['node', 'src/cli/index.ts', ...cliArgs];
const { main } = await import('./src/cli/index.ts');
await main();
`;

const runCli = (args) => {
    return new Promise((resolve, reject) => {
        const child = spawn(process.execPath, [
            '--input-type=module',
            '--import',
            'tsx',
            '-e',
            cliBootstrap
        ], {
            cwd: rootDir,
            env: {
                ...process.env,
                NO_COLOR: '1',
                [cliArgsEnvName]: JSON.stringify(args)
            },
            stdio: ['ignore', 'pipe', 'pipe']
        });

        let stdout = '';
        let stderr = '';
        child.stdout.setEncoding('utf8');
        child.stderr.setEncoding('utf8');
        child.stdout.on('data', chunk => {
            stdout += chunk;
        });
        child.stderr.on('data', chunk => {
            stderr += chunk;
        });
        child.on('error', reject);
        child.on('close', code => {
            resolve({ code, stdout, stderr });
        });
    });
};

describe('CLI parsing', () => {
    it('allows filter-sphere coordinates below their array indexes', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--filter-sphere',
            '0,1.6,0,15',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('allows negative filter-sphere center coordinates', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--filter-sphere',
            '-1,0,-2,10',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('accepts --lod -1 to tag an input as environment', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            'test/fixtures/splat/minimal.splat',
            '--lod',
            '-1',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('rejects --lod values below -1', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--lod',
            '-2',
            'null'
        ]);

        assert.notStrictEqual(result.code, 0, 'CLI should reject --lod -2');
        assert.match(result.stderr, /Must be >= 0, or -1/);
    });

    it('rejects an invalid --collision-color value', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-color',
            'bogus',
            'null'
        ]);

        assert.notStrictEqual(result.code, 0, 'CLI should reject --collision-color bogus');
        assert.match(result.stderr, /Invalid collision color mode: bogus\. Expected average or solid\./);
    });

    it('rejects the removed density-based modes', async () => {
        for (const mode of ['dominant', 'topk', 'gaussian']) {
            const result = await runCli([
                '--gpu',
                'cpu',
                'test/fixtures/splat/minimal.splat',
                '--collision-color',
                mode,
                'null'
            ]);

            assert.notStrictEqual(result.code, 0, `CLI should reject --collision-color ${mode}`);
            assert.match(result.stderr, /Invalid collision color mode.*Expected average or solid\./);
        }
    });

    it('accepts --collision-color case-insensitively', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-color',
            'SOLID',
            '--collision-mesh',
            'voxel',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        assert.doesNotMatch(result.stderr, /collision-color/i);
    });

    it('warns and ignores --collision-color without a collision mesh', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-color',
            'solid',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        assert.match(result.stderr, /--collision-color.*ignored/i);
    });

    it('warns and ignores --collision-color with a grey collision mesh shape', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-mesh',
            'faces',
            '--collision-color',
            'average',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        assert.match(result.stderr, /--collision-color.*ignored/i);
    });

    it('rejects --collision-color-palette with a non-integer value', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-color-palette',
            'notanumber',
            'null'
        ]);

        assert.notStrictEqual(result.code, 0, 'CLI should reject non-integer collision-color-palette');
    });

    it('rejects --collision-color-palette with value < 1', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-color-palette',
            '0',
            'null'
        ]);

        assert.notStrictEqual(result.code, 0, 'CLI should reject collision-color-palette < 1');
        assert.match(result.stderr, />= 1/);
    });

    it('warns and ignores --collision-color-palette without a collision mesh', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-color-palette',
            '8',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        assert.match(result.stderr, /--collision-color-palette.*ignored/i);
    });

    it('accepts a valid --collision-color-palette value', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-mesh',
            'voxel',
            '--collision-color-palette',
            '16',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        assert.doesNotMatch(result.stderr, /palette/i);
    });

    it('rejects a collision colour radius outside (0, 8]', async () => {
        for (const [flag, value] of [
            ['--collision-color-smooth', '0'],
            ['--collision-color-coherent', '9'],
            ['--collision-color-smooth', 'notanumber']
        ]) {
            const result = await runCli([
                '--gpu',
                'cpu',
                'test/fixtures/splat/minimal.splat',
                '--collision-mesh',
                'voxel',
                flag,
                value,
                'null'
            ]);

            assert.notStrictEqual(result.code, 0, `CLI should reject ${flag} ${value}`);
        }
    });

    it('warns and ignores collision colour radii without a coloured mesh', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-color-smooth',
            '2',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        assert.match(result.stderr, /--collision-color-smooth.*ignored/i);
    });

    it('accepts collision colour radii with a coloured mesh', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-mesh',
            'voxel',
            '--collision-color-palette',
            '8',
            '--collision-color-smooth',
            '1',
            '--collision-color-coherent',
            '1.5',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        assert.doesNotMatch(result.stderr, /ignored/i);
    });

    it('accepts --collision-color-flat with a coloured mesh', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--collision-mesh',
            'voxel',
            '--collision-color-flat',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });
});

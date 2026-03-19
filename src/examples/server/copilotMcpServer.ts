#!/usr/bin/env node

/**
 * GitHub Copilot MCP Server - Full Implementation
 *
 * Comprehensive MCP server integrating GitHub Copilot CLI with ALL MCP SDK features:
 * - Tools: Interactive Copilot commands
 * - Resources: Session history, logs, configuration
 * - Prompts: Common workflow templates
 * - Completions: Code and command suggestions
 */

import { spawn } from 'child_process';
import { z } from 'zod';
import { McpServer, ResourceTemplate } from '../../server/mcp.js';
import { StdioServerTransport } from '../../server/stdio.js';
import { CallToolResult } from '../../types.js';
import { appendFile, mkdir } from 'fs/promises';
import { existsSync, readFileSync, writeFileSync, mkdtempSync, rmSync, chmodSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';

const COPILOT_COMMAND = 'copilot';
const DEFAULT_TIMEOUT_MS = 60000;
const HELP_TIMEOUT_MS = 5000;
const DEFAULT_MAX_PROMPT_BYTES = 24000;

// Track temp dirs created for MCP config injection so we can clean up on exit
const mcpTmpDirs = new Set<string>();
const cleanupMcpTmpDirs = (): void => {
    for (const dir of mcpTmpDirs) {
        try { rmSync(dir, { recursive: true, force: true }); } catch (_) {}
    }
    mcpTmpDirs.clear();
};
process.on('exit', cleanupMcpTmpDirs);
process.on('SIGINT', () => { cleanupMcpTmpDirs(); process.exit(130); });
process.on('SIGTERM', () => { cleanupMcpTmpDirs(); process.exit(143); });

// Cache MCP config at startup to avoid repeated synchronous disk I/O per call
let cachedMcpConfigDir: string | undefined;
const initMcpConfigDir = (): void => {
    const mcpConfigPath = process.env.COPILOT_MCP_CONFIG_PATH?.trim()
        || join(homedir(), '.copilot', 'mcp-config.json');
    try {
        const raw = JSON.parse(readFileSync(mcpConfigPath, 'utf-8')) as {
            mcpServers: Record<string, unknown>;
        };

        // Filter servers: COPILOT_MCP_INCLUDE_SERVERS (whitelist) takes priority,
        // then COPILOT_MCP_EXCLUDE_SERVERS (blacklist), then inject all if neither set.
        const includeEnv = process.env.COPILOT_MCP_INCLUDE_SERVERS?.trim();
        const excludeEnv = process.env.COPILOT_MCP_EXCLUDE_SERVERS?.trim();
        let servers = raw.mcpServers ?? {};

        if (includeEnv) {
            const include = new Set(includeEnv.split(',').map(s => s.trim()));
            servers = Object.fromEntries(
                Object.entries(servers).filter(([k]) => include.has(k))
            );
        } else if (excludeEnv) {
            const exclude = new Set(excludeEnv.split(',').map(s => s.trim()));
            servers = Object.fromEntries(
                Object.entries(servers).filter(([k]) => !exclude.has(k))
            );
        }

        const filtered = JSON.stringify({ mcpServers: servers });
        const tmpDir = mkdtempSync(join(homedir(), '.copilot', 'mcp-tmp-'));
        chmodSync(tmpDir, 0o700);
        writeFileSync(join(tmpDir, 'mcp-config.json'), filtered, { mode: 0o600 });
        mcpTmpDirs.add(tmpDir);
        cachedMcpConfigDir = tmpDir;
        logMessage('debug', 'MCP config loaded', {
            path: mcpConfigPath,
            tmpDir,
            servers: Object.keys(servers)
        });
    } catch (e) {
        logMessage('debug', 'No MCP config found or failed to load', {
            path: mcpConfigPath,
            error: (e as Error).message
        });
    }
};
const FALLBACK_MODELS = [
    'claude-sonnet-4.5',
    'claude-haiku-4.5',
    'claude-opus-4.5',
    'claude-sonnet-4',
    'gpt-5.2-codex',
    'gpt-5.1-codex-max',
    'gpt-5.1-codex',
    'gpt-5.2',
    'gpt-5.1',
    'gpt-5',
    'gpt-5.1-codex-mini',
    'gpt-5-mini',
    'gpt-4.1',
    'gemini-3-pro-preview'
];
const MODEL_TOKEN_REGEX = /\b(?:claude|gpt|gemini|o)[a-z0-9.-]*\b/gi;
const COPILOT_NOT_INSTALLED_MESSAGE = 'GitHub Copilot CLI is not installed.\n\nInstall: npm install -g @github/copilot';

function parseBoolean(value: string | undefined): boolean | undefined {
    if (!value) {
        return undefined;
    }

    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'y', 'on'].includes(normalized)) {
        return true;
    }
    if (['false', '0', 'no', 'n', 'off'].includes(normalized)) {
        return false;
    }
    return undefined;
}

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVELS: Record<LogLevel, number> = {
    debug: 10,
    info: 20,
    warn: 30,
    error: 40
};

function normalizeLogLevel(value: string | undefined, debugEnabled: boolean): LogLevel {
    const normalized = value?.trim().toLowerCase();
    if (normalized === 'debug' || normalized === 'info' || normalized === 'warn' || normalized === 'error') {
        return normalized;
    }

    return debugEnabled ? 'debug' : 'info';
}

const DEBUG_ENABLED = parseBoolean(process.env.COPILOT_DEBUG) ?? false;
const LOG_LEVEL = normalizeLogLevel(process.env.COPILOT_LOG_LEVEL, DEBUG_ENABLED);
const LOG_FILE_PATH = process.env.COPILOT_LOG_FILE?.trim();

function shouldLog(level: LogLevel): boolean {
    return LOG_LEVELS[level] >= LOG_LEVELS[LOG_LEVEL];
}

function formatLogEntry(level: LogLevel, message: string, data?: Record<string, unknown>) {
    return {
        timestamp: new Date().toISOString(),
        level,
        message,
        ...(data ? { data } : {})
    };
}

function logMessage(level: LogLevel, message: string, data?: Record<string, unknown>) {
    if (!shouldLog(level)) {
        return;
    }

    const entry = formatLogEntry(level, message, data);

    if (LOG_FILE_PATH) {
        appendFile(LOG_FILE_PATH, `${JSON.stringify(entry)}\n`, 'utf8').catch(error => {
            if (DEBUG_ENABLED) {
                console.error(
                    `Warning: Failed to write log file ${LOG_FILE_PATH}: ${error instanceof Error ? error.message : String(error)}`
                );
            }
        });
    }

    if (DEBUG_ENABLED || level === 'warn' || level === 'error') {
        const payload = data ? ` ${JSON.stringify(data)}` : '';
        console.error(`[copilot-mcp] ${level}: ${message}${payload}`);
    }
}

function getDefaultModel(): string | undefined {
    const model = process.env.COPILOT_MODEL?.trim();
    return model ? model : undefined;
}

function getAllowAllToolsDefault(): boolean {
    return parseBoolean(process.env.COPILOT_ALLOW_ALL_TOOLS) ?? false;
}

function getTimeoutMs(): number {
    const timeoutValue = process.env.COPILOT_TIMEOUT;
    if (!timeoutValue) {
        return DEFAULT_TIMEOUT_MS;
    }

    const parsed = Number.parseInt(timeoutValue, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
        return parsed;
    }

    return DEFAULT_TIMEOUT_MS;
}

function getMaxPromptBytes(): number {
    const maxValue = process.env.COPILOT_MAX_PROMPT_BYTES;
    if (!maxValue) {
        return DEFAULT_MAX_PROMPT_BYTES;
    }

    const parsed = Number.parseInt(maxValue, 10);
    if (!Number.isFinite(parsed)) {
        return DEFAULT_MAX_PROMPT_BYTES;
    }

    if (parsed <= 0) {
        return Number.POSITIVE_INFINITY;
    }

    return parsed;
}

// Session management
interface CopilotSession {
    id: string;
    startTime: Date;
    lastActivity: Date;
    history: Array<{ prompt: string; response: string; timestamp: Date }>;
}

const sessions = new Map<string, CopilotSession>();
let currentSessionId: string | null = null;

// Paths
const copilotDir = join(homedir(), '.copilot');
const logsDir = join(copilotDir, 'logs');
const sessionsDir = join(copilotDir, 'mcp-sessions');

// Initialize directories
async function initDirectories() {
    for (const dir of [copilotDir, logsDir, sessionsDir]) {
        if (!existsSync(dir)) {
            try {
                await mkdir(dir, { recursive: true });
            } catch (error) {
                logMessage('warn', 'Failed to create directory', {
                    directory: dir,
                    error: error instanceof Error ? error.message : String(error)
                });
            }
        }
    }
}

// Check if Copilot CLI is installed
async function checkCopilotInstalled(): Promise<boolean> {
    return new Promise(resolve => {
        const child = spawn(COPILOT_COMMAND, ['--version'], {
            stdio: 'pipe'
        });

        child.on('error', () => resolve(false));
        child.on('exit', code => resolve(code === 0));

        setTimeout(() => {
            child.kill();
            resolve(false);
        }, 5000);
    });
}

async function getCopilotHelpOutput(): Promise<string> {
    return new Promise((resolve, reject) => {
        const child = spawn(COPILOT_COMMAND, ['--help'], {
            stdio: ['ignore', 'pipe', 'pipe']
        });

        let stdout = '';
        let stderr = '';
        let settled = false;
        let timeoutId: ReturnType<typeof setTimeout> | undefined;

        const finish = (fn: () => void) => {
            if (settled) {
                return;
            }
            settled = true;
            fn();
        };

        const clearTimeoutId = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        };

        child.stdout.on('data', data => {
            stdout += data.toString();
        });

        child.stderr.on('data', data => {
            stderr += data.toString();
        });

        child.on('error', error => {
            clearTimeoutId();
            finish(() => reject(new Error(`Failed to execute copilot --help: ${error.message}`)));
        });

        child.on('exit', () => {
            clearTimeoutId();
            const output = stdout.trim() || stderr.trim();
            if (!output) {
                finish(() => reject(new Error('No output from copilot --help')));
                return;
            }
            finish(() => resolve(output));
        });

        timeoutId = setTimeout(() => {
            child.kill();
            finish(() => reject(new Error('Copilot CLI help command timed out')));
        }, HELP_TIMEOUT_MS);
    });
}

function parseModelsFromHelpOutput(helpText: string): string[] {
    const matches = helpText.match(MODEL_TOKEN_REGEX) ?? [];
    const seen = new Set<string>();
    const models: string[] = [];

    for (const match of matches) {
        const token = match.toLowerCase();
        if (!/\d/.test(token)) {
            continue;
        }
        if (seen.has(token)) {
            continue;
        }
        seen.add(token);
        models.push(token);
    }

    return models;
}

async function listCopilotModels(): Promise<{ models: string[]; source: 'help' | 'fallback' }> {
    try {
        const helpOutput = await getCopilotHelpOutput();
        const parsed = parseModelsFromHelpOutput(helpOutput);
        if (parsed.length > 0) {
            return { models: parsed, source: 'help' };
        }
    } catch (error) {
        logMessage('warn', 'Failed to parse copilot --help', {
            error: error instanceof Error ? error.message : String(error)
        });
    }

    return { models: FALLBACK_MODELS, source: 'fallback' };
}

type CopilotCommandMeta = {
    model?: string;
    durationMs: number;
    exitCode?: number;
    signal?: string;
    stderrSnippet?: string;
    timedOut?: boolean;
};

type CopilotCommandResult = {
    text: string;
    meta: CopilotCommandMeta;
};

function getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function formatToolErrorMessage(message: string): string {
    return message.startsWith('Error:') ? message : `Error: ${message}`;
}

function truncateText(value: string, maxLength: number): string {
    const trimmed = value.trim();
    if (trimmed.length <= maxLength) {
        return trimmed;
    }

    return trimmed.slice(-maxLength);
}

function buildToolSuccess(result: CopilotCommandResult): CallToolResult {
    return {
        content: [{ type: 'text', text: result.text }],
        structuredContent: {
            text: result.text,
            ...result.meta
        }
    };
}

function buildToolError(message: string, meta?: Partial<CopilotCommandMeta>): CallToolResult {
    const formatted = formatToolErrorMessage(message);
    return {
        content: [{ type: 'text', text: formatted }],
        structuredContent: {
            error: message,
            ...(meta ?? {})
        },
        isError: true
    };
}

function buildCopilotNotInstalledError(): CallToolResult {
    return buildToolError(COPILOT_NOT_INSTALLED_MESSAGE);
}

// Execute Copilot CLI command with options
async function executeCopilotCommand(
    prompt: string,
    options: {
        context?: string;
        model?: string;
        allowAllTools?: boolean;
        sessionId?: string;
        additionalArgs?: string[];
    } = {}
): Promise<CopilotCommandResult> {
    return new Promise((resolve, reject) => {
        const fullPrompt = options.context ? `${prompt}\n\nContext:\n${options.context}` : prompt;
        const selectedModel = options.model ?? getDefaultModel();
        const allowAllTools = options.allowAllTools ?? getAllowAllToolsDefault();
        const timeoutMs = getTimeoutMs();
        const maxPromptBytes = getMaxPromptBytes();
        const startTime = Date.now();

        const promptBytes = Buffer.byteLength(fullPrompt, 'utf8');
        if (Number.isFinite(maxPromptBytes) && promptBytes > maxPromptBytes) {
            logMessage('warn', 'Prompt exceeds configured max length', {
                promptBytes,
                maxPromptBytes
            });
            reject(
                new Error(
                    `Prompt is ${promptBytes} bytes; exceeds COPILOT_MAX_PROMPT_BYTES (${maxPromptBytes}). ` +
                        'Reduce input size or increase the limit.'
                )
            );
            return;
        }

        const args: string[] = ['--silent'];

        // Add model selection
        if (selectedModel) {
            args.push('--model', selectedModel);
        }

        // Add tool permissions
        if (allowAllTools) {
            args.push('--allow-all-tools');
        }

        // Add session resume
        if (options.sessionId) {
            args.push('--resume', options.sessionId);
        }

        // Add additional arguments
        if (options.additionalArgs) {
            args.push(...options.additionalArgs);
        }

        // Inject user MCP config via a secure temp dir (chmod 700/600).
        // Using --config-dir avoids exposing secrets in argv (visible via ps/Task Manager).
        // The temp dir is created once at startup and cleaned up on process exit.
        if (cachedMcpConfigDir) {
            args.push('--config-dir', cachedMcpConfigDir);
        }

        const child = spawn(COPILOT_COMMAND, args, {
            stdio: ['pipe', 'pipe', 'pipe']
        });

        // Write prompt to stdin (avoids exposing prompt in process args)
        let stdout = '';
        let stderr = '';
        let hasReceivedOutput = false;
        let settled = false;
        let timeoutId: ReturnType<typeof setTimeout> | undefined;
        let exitCode: number | undefined;
        let signal: string | undefined;
        let timedOut = false;

        const finish = (fn: () => void) => {
            if (settled) {
                return;
            }
            settled = true;
            fn();
        };

        const clearTimeoutId = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        };

        // Handle stdin errors (e.g., EPIPE if process exits before reading)
        child.stdin.on('error', (error) => {
            logMessage('debug', 'Stdin write error (process may have exited early)', {
                error: error.message
            });
            // Don't reject here - let the 'exit' handler deal with the actual error
        });

        // Write prompt to stdin (avoids exposing prompt in process args)
        child.stdin.write(fullPrompt);
        child.stdin.end();

        logMessage('debug', 'Copilot command started', {
            model: selectedModel,
            allowAllTools,
            promptLength: fullPrompt.length,
            hasContext: Boolean(options.context)
        });

        child.stdout.on('data', data => {
            const output = data.toString();
            stdout += output;
            hasReceivedOutput = true;
        });

        child.stderr.on('data', data => {
            stderr += data.toString();
        });

        child.on('error', error => {
            clearTimeoutId();
            finish(() => reject(new Error(`Failed to execute copilot: ${error.message}`)));
        });

        child.on('exit', (code, exitSignal) => {
            clearTimeoutId();
            exitCode = code ?? undefined;
            signal = exitSignal ?? undefined;
            if (!hasReceivedOutput && stderr) {
                if (stderr.includes('login') || stderr.includes('authenticate')) {
                    finish(() => reject(new Error('GitHub Copilot CLI requires authentication. Please run: copilot /login')));
                    return;
                }
            }

            const resultText = stdout.trim() || stderr.trim() || 'No response from Copilot CLI';
            const durationMs = Date.now() - startTime;
            const stderrSnippet = stderr.trim() ? truncateText(stderr, 400) : undefined;
            const meta: CopilotCommandMeta = {
                model: selectedModel,
                durationMs,
                exitCode,
                signal,
                stderrSnippet,
                timedOut
            };

            // Save to session history
            if (currentSessionId && sessions.has(currentSessionId)) {
                const session = sessions.get(currentSessionId)!;
                session.history.push({
                    prompt: fullPrompt,
                    response: resultText,
                    timestamp: new Date()
                });
                session.lastActivity = new Date();
            }

            logMessage('info', 'Copilot command finished', {
                model: selectedModel,
                durationMs,
                exitCode,
                signal,
                timedOut
            });

            finish(() => resolve({ text: resultText, meta }));
        });

        // Timeout after 60 seconds
        timeoutId = setTimeout(() => {
            timedOut = true;
            child.kill();

            if (hasReceivedOutput) {
                const durationMs = Date.now() - startTime;
                const resultText = stdout.trim() || 'Copilot CLI timed out, but partial response received';
                const stderrSnippet = stderr.trim() ? truncateText(stderr, 400) : undefined;
                finish(() => resolve({
                    text: resultText,
                    meta: {
                        model: selectedModel,
                        durationMs,
                        exitCode,
                        signal,
                        stderrSnippet,
                        timedOut
                    }
                }));
            } else {
                finish(() => reject(new Error('Copilot CLI command timed out with no response')));
            }
        }, timeoutMs);
    });
}

// Create session
function createSession(): string {
    const sessionId = `session-${Date.now()}`;
    sessions.set(sessionId, {
        id: sessionId,
        startTime: new Date(),
        lastActivity: new Date(),
        history: []
    });
    currentSessionId = sessionId;
    return sessionId;
}

// Create the MCP server
const server = new McpServer({
    name: 'copilot-mcp-server',
    version: '2.0.0',
    description: 'Comprehensive GitHub Copilot CLI integration with full MCP capabilities'
});

// ============================================================================
// TOOLS - Interactive Copilot Commands
// ============================================================================

// 1. Ask Copilot (General purpose)
server.registerTool(
    'ask-copilot',
    {
        title: 'Ask GitHub Copilot',
        description: 'Ask GitHub Copilot CLI to help with coding tasks, generate commands, explain code, or provide suggestions',
        inputSchema: {
            prompt: z.string().describe('The question or task to ask GitHub Copilot CLI'),
            context: z.string().optional().describe('Optional additional context (file paths, code snippets, etc.)'),
            model: z.string().optional().describe('AI model to use'),
            allowAllTools: z.boolean().optional().describe('Allow all tools to run automatically without confirmation')
        }
    },
    async ({ prompt, context, model, allowAllTools }): Promise<CallToolResult> => {
        try {
            const isInstalled = await checkCopilotInstalled();
            if (!isInstalled) {
                return buildCopilotNotInstalledError();
            }

            const result = await executeCopilotCommand(prompt, { context, model, allowAllTools });
            return buildToolSuccess(result);
        } catch (error) {
            logMessage('error', 'Ask Copilot failed', { error: getErrorMessage(error) });
            return buildToolError(getErrorMessage(error));
        }
    }
);

// 2. Explain Code
server.registerTool(
    'copilot-explain',
    {
        title: 'Explain Code with Copilot',
        description: 'Get detailed explanations of code or technical concepts',
        inputSchema: {
            code: z.string().describe('The code or concept to explain'),
            model: z.string().optional().describe('AI model to use')
        }
    },
    async ({ code, model }): Promise<CallToolResult> => {
        try {
            const isInstalled = await checkCopilotInstalled();
            if (!isInstalled) {
                return buildCopilotNotInstalledError();
            }

            const result = await executeCopilotCommand(`Please explain this code:\n\n${code}`, { model });
            return buildToolSuccess(result);
        } catch (error) {
            logMessage('error', 'Copilot explain failed', { error: getErrorMessage(error) });
            return buildToolError(getErrorMessage(error));
        }
    }
);

// 3. Suggest Commands
server.registerTool(
    'copilot-suggest',
    {
        title: 'Get Command Suggestions',
        description: 'Get CLI command suggestions for specific tasks',
        inputSchema: {
            task: z.string().describe('The task you want to accomplish'),
            model: z.string().optional().describe('AI model to use')
        }
    },
    async ({ task, model }): Promise<CallToolResult> => {
        try {
            const isInstalled = await checkCopilotInstalled();
            if (!isInstalled) {
                return buildCopilotNotInstalledError();
            }

            const result = await executeCopilotCommand(`Suggest a command for: ${task}`, { model });
            return buildToolSuccess(result);
        } catch (error) {
            logMessage('error', 'Copilot suggest failed', { error: getErrorMessage(error) });
            return buildToolError(getErrorMessage(error));
        }
    }
);

// 4. List Models
server.registerTool(
    'copilot-list-models',
    {
        title: 'List Copilot Models',
        description: 'List available GitHub Copilot CLI model identifiers',
        inputSchema: {}
    },
    async (): Promise<CallToolResult> => {
        try {
            const isInstalled = await checkCopilotInstalled();
            if (!isInstalled) {
                return buildCopilotNotInstalledError();
            }

            const { models, source } = await listCopilotModels();
            const header = source === 'help'
                ? 'Available Copilot models (from copilot --help):'
                : 'Available Copilot models (fallback list):';
            const body = models.map(model => `- ${model}`).join('\n');
            const text = `${header}\n${body}`;

            return {
                content: [{ type: 'text', text }],
                structuredContent: { models, source }
            };
        } catch (error) {
            logMessage('error', 'Copilot list models failed', { error: getErrorMessage(error) });
            return buildToolError(getErrorMessage(error));
        }
    }
);

// 5. Debug Code
server.registerTool(
    'copilot-debug',
    {
        title: 'Debug Code',
        description: 'Help debug code errors and issues',
        inputSchema: {
            code: z.string().describe('The code with the error'),
            error: z.string().describe('The error message or description'),
            context: z.string().optional().describe('Additional context about the error')
        }
    },
    async ({ code, error, context }): Promise<CallToolResult> => {
        try {
            const isInstalled = await checkCopilotInstalled();
            if (!isInstalled) {
                return buildCopilotNotInstalledError();
            }

            const prompt = `Debug this code:\n\n${code}\n\nError: ${error}`;
            const result = await executeCopilotCommand(prompt, { context });
            return buildToolSuccess(result);
        } catch (error) {
            logMessage('error', 'Copilot debug failed', { error: getErrorMessage(error) });
            return buildToolError(getErrorMessage(error));
        }
    }
);

// 6. Refactor Code
server.registerTool(
    'copilot-refactor',
    {
        title: 'Refactor Code',
        description: 'Suggest refactoring improvements for code',
        inputSchema: {
            code: z.string().describe('The code to refactor'),
            goal: z.string().optional().describe('Specific refactoring goal (e.g., "improve performance", "increase readability")')
        }
    },
    async ({ code, goal }): Promise<CallToolResult> => {
        try {
            const isInstalled = await checkCopilotInstalled();
            if (!isInstalled) {
                return buildCopilotNotInstalledError();
            }

            const prompt = goal
                ? `Refactor this code to ${goal}:\n\n${code}`
                : `Refactor and improve this code:\n\n${code}`;
            const result = await executeCopilotCommand(prompt);
            return buildToolSuccess(result);
        } catch (error) {
            logMessage('error', 'Copilot refactor failed', { error: getErrorMessage(error) });
            return buildToolError(getErrorMessage(error));
        }
    }
);

// 7. Generate Tests
server.registerTool(
    'copilot-test-generate',
    {
        title: 'Generate Tests',
        description: 'Generate unit tests for code',
        inputSchema: {
            code: z.string().describe('The code to generate tests for'),
            framework: z.string().optional().describe('Testing framework to use (e.g., Jest, Mocha, pytest)')
        }
    },
    async ({ code, framework }): Promise<CallToolResult> => {
        try {
            const isInstalled = await checkCopilotInstalled();
            if (!isInstalled) {
                return buildCopilotNotInstalledError();
            }

            const prompt = framework
                ? `Generate ${framework} tests for this code:\n\n${code}`
                : `Generate unit tests for this code:\n\n${code}`;
            const result = await executeCopilotCommand(prompt);
            return buildToolSuccess(result);
        } catch (error) {
            logMessage('error', 'Copilot test generate failed', { error: getErrorMessage(error) });
            return buildToolError(getErrorMessage(error));
        }
    }
);

// 8. Review Code
server.registerTool(
    'copilot-review',
    {
        title: 'Review Code',
        description: 'Get a code review with suggestions',
        inputSchema: {
            code: z.string().describe('The code to review'),
            focusAreas: z.array(z.string()).optional().describe('Specific areas to focus on (e.g., ["security", "performance"])')
        }
    },
    async ({ code, focusAreas }): Promise<CallToolResult> => {
        try {
            const isInstalled = await checkCopilotInstalled();
            if (!isInstalled) {
                return buildCopilotNotInstalledError();
            }

            const prompt = focusAreas && focusAreas.length > 0
                ? `Review this code, focusing on ${focusAreas.join(', ')}:\n\n${code}`
                : `Review this code:\n\n${code}`;
            const result = await executeCopilotCommand(prompt);
            return buildToolSuccess(result);
        } catch (error) {
            logMessage('error', 'Copilot review failed', { error: getErrorMessage(error) });
            return buildToolError(getErrorMessage(error));
        }
    }
);

// 9. Session Management
server.registerTool(
    'copilot-session-start',
    {
        title: 'Start New Session',
        description: 'Start a new Copilot conversation session',
        inputSchema: {}
    },
    async (): Promise<CallToolResult> => {
        const sessionId = createSession();
        return {
            content: [{
                type: 'text',
                text: `New session started: ${sessionId}\nAll subsequent interactions will be tracked in this session.`
            }]
        };
    }
);

server.registerTool(
    'copilot-session-history',
    {
        title: 'Get Session History',
        description: 'Retrieve the conversation history for the current session',
        inputSchema: {
            sessionId: z.string().optional().describe('Session ID (defaults to current session)')
        }
    },
    async ({ sessionId }): Promise<CallToolResult> => {
        const targetId = sessionId || currentSessionId;
        if (!targetId || !sessions.has(targetId)) {
            return {
                content: [{ type: 'text', text: 'No session found. Start a new session with copilot-session-start.' }],
                isError: true
            };
        }

        const session = sessions.get(targetId)!;
        const historyText = session.history.map((entry, i) =>
            `\n[${i + 1}] ${entry.timestamp.toISOString()}\nPrompt: ${entry.prompt}\nResponse: ${entry.response.substring(0, 200)}...`
        ).join('\n');

        return {
            content: [{
                type: 'text',
                text: `Session: ${session.id}\nStarted: ${session.startTime.toISOString()}\nLast Activity: ${session.lastActivity.toISOString()}\n\nHistory:${historyText || ' (empty)'}`
            }]
        };
    }
);

// ============================================================================
// RESOURCES - Session History, Logs, Configuration
// ============================================================================

// Resource: Session history
server.registerResource(
    'session-history',
    new ResourceTemplate('copilot://session/{sessionId}/history', { list: undefined }),
    {
        title: 'Copilot Session History',
        description: 'Access conversation history for a specific session'
    },
    async (uri, { sessionId }) => {
        if (!sessions.has(sessionId as string)) {
            throw new Error(`Session not found: ${sessionId}`);
        }

        const session = sessions.get(sessionId as string)!;
        const historyJson = JSON.stringify(session, null, 2);

        return {
            contents: [{
                uri: uri.href,
                text: historyJson,
                mimeType: 'application/json'
            }]
        };
    }
);

// Resource: All sessions list
server.registerResource(
    'sessions-list',
    new ResourceTemplate('copilot://sessions', { list: undefined }),
    {
        title: 'All Copilot Sessions',
        description: 'List all active Copilot sessions'
    },
    async (uri) => {
        const sessionsList = Array.from(sessions.values()).map(s => ({
            id: s.id,
            startTime: s.startTime,
            lastActivity: s.lastActivity,
            messageCount: s.history.length
        }));

        return {
            contents: [{
                uri: uri.href,
                text: JSON.stringify(sessionsList, null, 2),
                mimeType: 'application/json'
            }]
        };
    }
);

// ============================================================================
// PROMPTS - Common Workflow Templates
// ============================================================================

// Prompt: Code review template
server.registerPrompt(
    'code-review-template',
    {
        title: 'Code Review Template',
        description: 'A structured template for conducting code reviews',
        argsSchema: {
            code: z.string().describe('The code to review'),
            language: z.string().optional().describe('Programming language')
        }
    },
    async ({ code, language }) => ({
        messages: [
            {
                role: 'user',
                content: {
                    type: 'text',
                    text: `Please review the following ${language || ''} code and provide feedback on:
1. Code quality and style
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Best practices

Code:
\`\`\`${language || ''}
${code}
\`\`\`

Provide specific, actionable feedback.`
                }
            }
        ]
    })
);

// Prompt: Debug assistance template
server.registerPrompt(
    'debug-template',
    {
        title: 'Debug Assistance Template',
        description: 'A structured template for debugging assistance',
        argsSchema: {
            code: z.string().describe('The code with the error'),
            error: z.string().describe('The error message'),
            context: z.string().optional().describe('Additional context')
        }
    },
    async ({ code, error, context }) => ({
        messages: [
            {
                role: 'user',
                content: {
                    type: 'text',
                    text: `I'm getting an error and need help debugging:

Error: ${error}

Code:
\`\`\`
${code}
\`\`\`

${context ? `Context: ${context}` : ''}

Please help me:
1. Identify the root cause
2. Explain why the error is happening
3. Provide a fix
4. Suggest how to prevent similar errors`
                }
            }
        ]
    })
);

// Prompt: Refactoring template
server.registerPrompt(
    'refactor-template',
    {
        title: 'Refactoring Template',
        description: 'A template for requesting code refactoring',
        argsSchema: {
            code: z.string().describe('The code to refactor'),
            goal: z.string().optional().describe('Refactoring goal')
        }
    },
    async ({ code, goal }) => ({
        messages: [
            {
                role: 'user',
                content: {
                    type: 'text',
                    text: `Please refactor this code${goal ? ` to ${goal}` : ''}:

\`\`\`
${code}
\`\`\`

Provide:
1. Refactored code
2. Explanation of changes
3. Benefits of the refactoring
4. Any trade-offs or considerations`
                }
            }
        ]
    })
);

// ============================================================================
// SERVER INITIALIZATION
// ============================================================================

async function main() {
    console.error('🚀 Starting GitHub Copilot MCP Server (Full Implementation)...');

    // Initialize directories
    await initDirectories();

    // Load MCP config into a secure temp dir for subprocess injection
    initMcpConfigDir();

    // Create initial session
    createSession();
    console.error(`✅ Initial session created: ${currentSessionId}`);

    const transport = new StdioServerTransport();
    await server.connect(transport);

    console.error('✅ Server running on stdio');
    console.error('📦 Features enabled:');
    console.error('   - 9 Tools (ask, explain, suggest, list-models, debug, refactor, test, review, session)');
    console.error('   - 2 Resources (session history, sessions list)');
    console.error('   - 3 Prompts (code review, debug, refactor templates)');
    console.error('Waiting for requests...\n');
}

main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});

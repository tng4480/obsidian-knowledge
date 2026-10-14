<#
Sets up Codex to use this project: the academic_memory MCP server entry in ~/.codex/config.toml and the guidance in
~/.codex/AGENTS.md. Safe to run again: the guidance is kept between marker comments and replaced in place (anything else
in AGENTS.md is untouched), and an existing server entry is never overwritten.

    powershell -ExecutionPolicy Bypass -File codex\install.ps1 -Vault 'C:\path\to\vault'

Needs the project's virtual environment first (`uv sync`). Restart Codex afterwards.
#>
param(
    [string]$Vault = $env:ACADMEM_VAULT
)
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)

$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "No virtual environment at $python. Run 'uv sync' in $repo first." }
if (-not $Vault) { throw "Pass the vault folder: -Vault 'C:\path\to\vault' (or set ACADMEM_VAULT)." }
if (-not (Test-Path -LiteralPath $Vault -PathType Container)) { throw "Vault folder not found: $Vault" }
if ($Vault.Contains("'") -or $python.Contains("'")) { throw "Paths containing a single quote are not supported." }

$codex = Join-Path $HOME '.codex'
[void][IO.Directory]::CreateDirectory($codex)

# --- AGENTS.md: the guidance block lives between two marker comments -------------------------------------------------
$begin = '<!-- acadmem:begin (managed by codex/install.ps1 in the obsidian-knowledge repo; edit it there) -->'
$end = '<!-- acadmem:end -->'
$guidance = ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'AGENTS.md'), $utf8)).Trim()
$block = "$begin`r`n$guidance`r`n$end"
$agents = Join-Path $codex 'AGENTS.md'
$text = if (Test-Path -LiteralPath $agents) { [IO.File]::ReadAllText($agents, $utf8) } else { '' }
$pattern = '(?s)<!-- acadmem:begin.*?<!-- acadmem:end -->'
if ($text -match $pattern) {
    $text = [regex]::Replace($text, $pattern, { param($m) $block })
    $action = 'updated'
} else {
    if (Test-Path -LiteralPath $agents) { Copy-Item -LiteralPath $agents "$agents.bak-before-acadmem" -Force }
    $text = if ($text.Trim()) { $text.TrimEnd() + "`r`n`r`n" + $block + "`r`n" } else { $block + "`r`n" }
    $action = 'added'
}
[IO.File]::WriteAllText($agents, $text, $utf8)
Write-Host "AGENTS.md: guidance $action ($agents)"

# --- config.toml: the server entry ------------------------------------------------------------------------------------
$config = Join-Path $codex 'config.toml'
$toml = if (Test-Path -LiteralPath $config) { [IO.File]::ReadAllText($config, $utf8) } else { '' }
if ($toml -match '\[mcp_servers\.academic_memory\]') {
    Write-Host "config.toml: [mcp_servers.academic_memory] already exists, left alone. Check its paths: python = $python, vault = $Vault"
} else {
    $entry = ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'config.example.toml'), $utf8)).Replace('@@PYTHON@@', $python).Replace('@@VAULT@@', $Vault)
    if (Test-Path -LiteralPath $config) { Copy-Item -LiteralPath $config "$config.bak-before-acadmem" -Force }
    [IO.File]::WriteAllText($config, $toml.TrimEnd() + "`r`n`r`n" + $entry, $utf8)
    Write-Host "config.toml: academic_memory server added ($config)"
}
Write-Host 'Done. Restart Codex so it picks up the changes.'

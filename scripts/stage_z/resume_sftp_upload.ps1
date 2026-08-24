param(
    [Parameter(Mandatory)] [string] $LocalPath,
    [Parameter(Mandatory)] [string] $RemotePath,
    [Parameter(Mandatory)] [long] $ExpectedBytes,
    [int] $MaxAttempts = 100
)

$sshKey = Join-Path $env:USERPROFILE '.ssh\id_ed25519_vla'
$localForSftp = $LocalPath.Replace('\', '/')

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    @("reput `"$localForSftp`" `"$RemotePath`"") |
        & sftp.exe -c aes128-ctr -b - -B 16777216 -R 256 -o ProxyJump=none -o ConnectionAttempts=2 -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -P 33571 -i $sshKey dty_user@10.60.2.56 | Out-Null
    $sftpExit = $LASTEXITCODE

    $sizeText = & ssh.exe -c aes128-ctr -o ProxyJump=none -o ConnectionAttempts=1 -o ConnectTimeout=8 -i $sshKey -p 33571 dty_user@10.60.2.56 "stat -c '%s' '$RemotePath' 2>/dev/null || echo 0" 2>$null
    $size = 0L
    [long]::TryParse(($sizeText | Select-Object -Last 1), [ref]$size) | Out-Null
    Write-Output (ConvertTo-Json @{attempt=$attempt; size=$size; expected=$ExpectedBytes; sftp_exit=$sftpExit} -Compress)

    if ($size -eq $ExpectedBytes) { exit 0 }
    if ($size -gt $ExpectedBytes) { exit 2 }
    Start-Sleep -Seconds 2
}

exit 1

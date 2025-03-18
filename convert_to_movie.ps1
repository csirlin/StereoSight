# Path to ffmpeg executable (adjust if necessary)
$ffmpegPath = ".\ffmpeg.exe"

# Duration of the output video
$duration = 15

# Loop through each PNG file in the current directory
Get-ChildItem -Filter output\png\*.png | ForEach-Object {
    $inputFile = $_.FullName
    $outputFile = "output\mp4\$($_.BaseName).mp4"

    # Run ffmpeg command
    & $ffmpegPath -n -loop 1 -i $inputFile -t $duration -vf "format=yuv420p" $outputFile
}
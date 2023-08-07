using CommandLine;
// ReSharper disable ClassNeverInstantiated.Global
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.

namespace MissingRustTwitchDrops;

internal class Options
{
    [Option('d', "directories", Required = false, HelpText = "Directories to process.")]
    public IEnumerable<string>? Directories { get; set; }

    [Option('o', "output", Required = false, HelpText = "Output directory.")]
    public string? OutputDirectory { get; set; }
}
using System.Reflection;
using CommandLine;
using Emgu.CV;

namespace MissingRustTwitchDrops;

internal static class Program
{
    private static void Main(string[] args)
    {
        Environment.SetEnvironmentVariable("EMGU_CV_RUNTIME_DIR",
            Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location));

        Parser.Default.ParseArguments<Options>(args)
            .WithParsed(RunOptionsAndReturnExitCode)
            .WithNotParsed(HandleParseError);
    }

    private static void RunOptionsAndReturnExitCode(Options opts)
    {
        var directories = GetDirectories(opts);
        var outputDirectory = GetOutputDirectory(opts);

        Console.WriteLine($"Processing images in {directories[0]} and {directories[1]}...");
        var images1 = ProcessImagesInDirectory(directories[0]);
        var images2 = ProcessImagesInDirectory(directories[1]);

        Console.WriteLine("Comparing images...");
        var onlyInDir1 = CompareImages(images1, images2);
        var onlyInDir2 = CompareImages(images2, images1);

        var onlyInDir1ValuePairs = onlyInDir1.ToList();
        Console.WriteLine($"Only in dir 1: {onlyInDir1ValuePairs.Count}");

        var onlyInDir2ValuePairs = onlyInDir2.ToList();
        Console.WriteLine($"Only in dir 2: {onlyInDir2ValuePairs.Count}");
        
        CombineAndSaveImages(onlyInDir1ValuePairs, Path.Combine(outputDirectory, "output1.png"));
        CombineAndSaveImages(onlyInDir2ValuePairs, Path.Combine(outputDirectory, "output2.png"));

        DisposeMatList(images1);
        DisposeMatList(images2);
    }

    private static void DisposeMatList(List<KeyValuePair<Mat, string>> images)
    {
        foreach (var kvp in images)
        {
            kvp.Key.Dispose();
        }
    }

    private static void HandleParseError(IEnumerable<Error> errs)
    {
        Console.WriteLine("Failed to parse command line arguments:");
        foreach (var err in errs)
        {
            Console.WriteLine(err);
        }
    }

    private static List<string> GetDirectories(Options opts)
    {
        var directories = opts.Directories?.ToList();
        if (directories?.Count < 2)
        {
            directories = Directory.GetDirectories(Directory.GetCurrentDirectory()).Take(2).ToList();
        }

        return directories!;
    }

    private static string GetOutputDirectory(Options opts)
    {
        return opts.OutputDirectory ?? Directory.GetCurrentDirectory();
    }

    private static List<KeyValuePair<Mat, string>> ProcessImagesInDirectory(string directory)
    {
        Console.WriteLine($"Processing images in {directory}...");
        return ImageProcessor.ProcessDirectory(directory);
    }

    private static IEnumerable<KeyValuePair<Mat, string>> CompareImages(IEnumerable<KeyValuePair<Mat, string>> images1,
        List<KeyValuePair<Mat, string>> images2)
    {
        return ImageComparator.Compare(images1, images2);
    }

    private static void CombineAndSaveImages(IEnumerable<KeyValuePair<Mat, string>> images, string outputFile)
    {
        Console.WriteLine($"Combining images and saving to {outputFile}...");
        CombineImages(images, outputFile);
    }

    private static void CombineImages(IEnumerable<KeyValuePair<Mat, string>> images, string outputFile)
    {
        var imageList = images.Select(x => x.Key).ToList();

        if (!imageList.Any())
        {
            Console.WriteLine("No images to combine.");
            return;
        }

        var imageCount = imageList.Count;
        var gridSize = (int)Math.Ceiling(Math.Sqrt(imageCount));

        var rows = new List<Mat>();
        for (var i = 0; i < gridSize; i++)
        {
            var rowImages = new List<Mat>();
            for (var j = 0; j < gridSize; j++)
            {
                var index = i * gridSize + j;
                var img = index < imageCount
                    ? imageList[index]
                    : // If there aren't enough images to fill the grid, fill the rest with blank images
                    Mat.Zeros(imageList[0].Size.Height, imageList[0].Size.Width, imageList[0].Depth,
                        imageList[0].NumberOfChannels);
                
                // resize the image to match the first image size
                CvInvoke.Resize(img, img, imageList[0].Size);
                
                rowImages.Add(img);
            }

            // Concatenate the images in the row horizontally
            var row = new Mat();
            CvInvoke.HConcat(rowImages.ToArray(), row);
            rows.Add(row);
        }

        // Concatenate the rows vertically to get the final image
        var grid = new Mat();
        CvInvoke.VConcat(rows.ToArray(), grid);

        // Save the combined image
        Console.WriteLine($"Saving combined image to {outputFile}...");
        
        if(File.Exists(outputFile))
            File.Delete(outputFile);
        
        grid.Save(outputFile);
    }
}
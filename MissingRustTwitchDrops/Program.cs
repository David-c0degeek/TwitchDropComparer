using System.Drawing;
using System.Reflection;
using CommandLine;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace MissingRustTwitchDrops;

internal static class Program
{
    private static void Main(string[] args)
    {
        Environment.SetEnvironmentVariable("EMGU_CV_RUNTIME_DIR", Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location));

        Parser.Default.ParseArguments<Options>(args)
            .WithParsed(RunOptionsAndReturnExitCode)
            .WithNotParsed(HandleParseError);
    }

    private static void RunOptionsAndReturnExitCode(Options opts)
    {
        var imageProcessor = new ImageProcessor();

        var directories = GetDirectories(opts);
        var outputDirectory = GetOutputDirectory(opts);

        Console.WriteLine($"Processing images in {directories[0]} and {directories[1]}...");
        var images1 = ProcessImagesInDirectory(imageProcessor, directories[0]);
        var images2 = ProcessImagesInDirectory(imageProcessor, directories[1]);

        Console.WriteLine("Comparing images...");
        var onlyInDir1 = CompareImages(images1, images2);
        var onlyInDir2 = CompareImages(images2, images1);

        PrintMissingImages("Dir1", onlyInDir1);
        PrintMissingImages("Dir2", onlyInDir2);

        CombineAndSaveImages(onlyInDir1, Path.Combine(outputDirectory, "output1.png"));
        CombineAndSaveImages(onlyInDir2, Path.Combine(outputDirectory, "output2.png"));
        
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

    private static List<KeyValuePair<Mat, string>> ProcessImagesInDirectory(ImageProcessor imageProcessor, string directory)
    {
        Console.WriteLine($"Processing images in {directory}...");
        return ImageProcessor.ProcessDirectory(directory);
    }

    private static List<KeyValuePair<Mat, string>> CompareImages(List<KeyValuePair<Mat, string>> images1, List<KeyValuePair<Mat, string>> images2)
    {
        return ImageComparator.Compare(images1, images2);
    }

    private static void PrintMissingImages(string dir, IEnumerable<KeyValuePair<Mat, string>> missingImages)
    {
        Console.WriteLine($"Only in {dir}:");
        foreach (var missingImage in missingImages)
        {
            Console.WriteLine($"{missingImage.Key} in {missingImage.Value}");
        }
    }

    private static void CombineAndSaveImages(IEnumerable<KeyValuePair<Mat, string>> images, string outputFile)
    {
        Console.WriteLine($"Combining images and saving to {outputFile}...");
        CombineImages(images, outputFile);
    }

    private static void CombineImages(IEnumerable<KeyValuePair<Mat, string>> images, string outputFile)
    {
        var valuePairs = images.ToList();
        if (!valuePairs.Any())
        {
            Console.WriteLine("No images to combine.");
            return;
        }
        
        // Compute the total width and maximum height of the combined image
        int totalWidth = 0, maxHeight = 0;
        var keyValuePairs = valuePairs.ToList();
        foreach (var image in keyValuePairs)
        {
            totalWidth += image.Key.Width;
            maxHeight = Math.Max(maxHeight, image.Key.Height);
        }

        // Create a new image with the computed size
        var combinedImage = new Mat(maxHeight, totalWidth, DepthType.Cv8U, 3);

        // Copy each image into the combined image
        var currentX = 0;
        foreach (var image in keyValuePairs)
        {
            var roi = new Mat(combinedImage, new Rectangle(new Point(currentX, 0), image.Key.Size));
            image.Key.CopyTo(roi);
            currentX += image.Key.Width;
        }

        // Save the combined image
        combinedImage.Save(outputFile);
    }
}
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace MissingRustTwitchDrops;

internal static class Program
{
    private static void Main()
    {
        var imageProcessor = new ImageProcessor();
        var imageComparator = new ImageComparator();

        // Directory paths
        var dir1 = "/path/to/dir1";
        var dir2 = "/path/to/dir2";

        // Extract smaller images from larger images and compute their descriptors
        var images1 = imageProcessor.ProcessDirectory(dir1);
        var images2 = imageProcessor.ProcessDirectory(dir2);

        // Compare the descriptors to find images that are only in one directory
        var onlyInDir1 = imageComparator.Compare(images1, images2);
        var onlyInDir2 = imageComparator.Compare(images2, images1);

        PrintMissingImages("Dir1", onlyInDir1);
        PrintMissingImages("Dir2", onlyInDir2);

        CombineImages(onlyInDir1, "/path/to/output1.png");
        CombineImages(onlyInDir2, "/path/to/output2.png");
    }

    private static void PrintMissingImages(string dir, IEnumerable<KeyValuePair<Mat, string>> missingImages)
    {
        Console.WriteLine($"Only in {dir}:");
        foreach (var missingImage in missingImages)
        {
            Console.WriteLine($"{missingImage.Key} in {missingImage.Value}");
        }
    }

    private static void CombineImages(IEnumerable<KeyValuePair<Mat, string>> images, string outputFile)
    {
        // Compute the total width and maximum height of the combined image
        int totalWidth = 0, maxHeight = 0;
        foreach (var image in images)
        {
            totalWidth += image.Key.Width;
            maxHeight = Math.Max(maxHeight, image.Key.Height);
        }

        // Create a new image with the computed size
        var combinedImage = new Mat(maxHeight, totalWidth, DepthType.Cv8U, 3);

        // Copy each image into the combined image
        var currentX = 0;
        foreach (var image in images)
        {
            var roi = new Mat(combinedImage, new Rectangle(new Point(currentX, 0), image.Key.Size));
            image.Key.CopyTo(roi);
            currentX += image.Key.Width;
        }

        // Save the combined image
        combinedImage.Save(outputFile);
    }
}
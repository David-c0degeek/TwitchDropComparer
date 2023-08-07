using Emgu.CV;
using System.Drawing;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace MissingRustTwitchDrops;

internal static class ImageProcessor
{
    private const int GridWidth = 5;
    private const int GridHeight = 5;

    private static readonly List<Bgr> ColorsToFilter = new()
    {
        new Bgr(62, 65, 66),  // #42413E
        new Bgr(241, 163, 53),  // #35A3F1
        new Bgr(64, 88, 241),  // #F15840
        new Bgr(0, 0, 0)  // #000000
    };
    
    public static List<KeyValuePair<Mat, string>> ProcessDirectory(string directory)
    {
        var imageFiles = Directory.GetFiles(directory, "*.png");
        var images = new List<KeyValuePair<Mat, string>>();
        foreach (var imageFile in imageFiles)
        {
            var image = CvInvoke.Imread(imageFile, ImreadModes.AnyColor);
            var gridImages = ExtractGridImages(image);
            images.AddRange(gridImages.Select(gridImage => new KeyValuePair<Mat, string>(gridImage, imageFile)));
        }
        return images;
    }

    /// <summary>
    /// Main bg: #42413E
    /// Outline 1: #35A3F1
    /// Outline 2: #F15840
    /// Between grid boxes: #000000
    /// </summary>
    /// <param name="image"></param>
    /// <param name="colorThreshold"></param>
    /// <returns></returns>
    public static Mat FilterOutColors(Mat image, double colorThreshold = 30.0)
    {
        // Create a copy of the image
        var filteredImage = image.Clone();

        // Convert the image to the Bgr color space
        if (filteredImage.NumberOfChannels == 1)
        {
            CvInvoke.CvtColor(filteredImage, filteredImage, ColorConversion.Gray2Bgr);
        }

        // Loop over the colors to filter
        foreach (var color in ColorsToFilter)
        {
            // Create a mask for the current color
            var lowerBound = new ScalarArray(new MCvScalar(color.Blue - colorThreshold, color.Green - colorThreshold, color.Red - colorThreshold));
            var upperBound = new ScalarArray(new MCvScalar(color.Blue + colorThreshold, color.Green + colorThreshold, color.Red + colorThreshold));
            var mask = new Mat();
            CvInvoke.InRange(filteredImage, lowerBound, upperBound, mask);

            // Apply the mask to the image
            filteredImage.SetTo(new MCvScalar(0, 0, 0), mask);
        }

        return filteredImage;
    }
    
    private static Mat ReduceNoise(Mat originalImage)
    {
        // Convert the image to the HSV color space
        var hsvImage = new Mat();
        CvInvoke.CvtColor(originalImage, hsvImage, ColorConversion.Bgr2Hsv);

        // Define the lower and upper bounds of the HSV color range for the background and grid lines
        // NOTE: You will need to adjust these values to match your specific images
        var lowerBound = new ScalarArray(new MCvScalar(0, 0, 0));
        var upperBound = new ScalarArray(new MCvScalar(180, 255, 255));

        // Create a binary mask of the pixels within the color range
        var mask = new Mat();
        CvInvoke.InRange(hsvImage, lowerBound, upperBound, mask);

        // Invert the mask
        var invertedMask = new Mat();
        CvInvoke.BitwiseNot(mask, invertedMask);

        // Apply the mask to the original image
        var result = new Mat();
        CvInvoke.BitwiseAnd(originalImage, originalImage, result, invertedMask);

        return result;
    }

    private static IEnumerable<Mat> ExtractGridImages(Mat image)
    {
        var cellWidth = image.Width / GridWidth;
        var cellHeight = image.Height / GridHeight;

        for (var y = 0; y < GridHeight; y++)
        {
            for (var x = 0; x < GridWidth; x++)
            {
                var cellRect = new Rectangle(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
                var cellImage = new Mat(image, cellRect);

                yield return cellImage;
            }
        }
    }
}
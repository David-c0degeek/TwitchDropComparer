using Emgu.CV;
using System.Drawing;

namespace MissingRustTwitchDrops;

internal class ImageProcessor
{
    private const int GridWidth = 5;
    private const int GridHeight = 5;

    public static List<KeyValuePair<Mat, string>> ProcessDirectory(string dir)
    {
        var images = new List<KeyValuePair<Mat, string>>();
        foreach (var file in Directory.GetFiles(dir))
        {
            images.AddRange(ExtractGridImages(file).Select(gridImage => new KeyValuePair<Mat, string>(gridImage, file)));
        }
        return images;
    }

    private static IEnumerable<Mat> ExtractGridImages(string file)
    {
        using var image = new Mat(file);

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
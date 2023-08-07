using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace MissingRustTwitchDrops;

internal class ImageProcessor
{
    public List<KeyValuePair<Mat, string>> ProcessDirectory(string dir)
    {
        var images = new List<KeyValuePair<Mat, string>>();
        foreach (var file in Directory.GetFiles(dir))
        {
            images.AddRange(ExtractSmallerImages(file).Select(smallerImage => new KeyValuePair<Mat, string>(smallerImage, file)));
        }
        return images;
    }

    private static IEnumerable<Mat> ExtractSmallerImages(string file)
    {
        using var src = new Mat(file, ImreadModes.Color);
        using var gray = new Mat();
        using var edges = new Mat();

        // Convert to grayscale
        CvInvoke.CvtColor(src, gray, ColorConversion.Bgr2Gray);

        // Detect edges
        CvInvoke.Canny(gray, edges, 50, 150);

        // Find contours
        using var contours = new VectorOfVectorOfPoint();
        CvInvoke.FindContours(edges, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

        // Extract each contour (smaller image) as a separate Mat
        for (var i = 0; i < contours.Size; i++)
        {
            using var contour = new VectorOfPoint(contours[i].ToArray());
            var boundingRect = CvInvoke.BoundingRectangle(contour);
            using var smallerImg = new Mat(src, boundingRect);

            yield return smallerImg;
        }
    }

}
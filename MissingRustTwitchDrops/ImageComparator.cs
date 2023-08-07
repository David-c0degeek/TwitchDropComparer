using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace MissingRustTwitchDrops;

internal class ImageComparator
{
    public static List<KeyValuePair<Mat, string>> Compare(IEnumerable<KeyValuePair<Mat, string>> images1, List<KeyValuePair<Mat, string>> images2)
    {
        return (
                from image1 in images1
                let matchFound = images2
                    .Any(image2 =>
                        CompareImages(image1.Key, image2.Key))
                where !matchFound
                select image1)
            .ToList();
    }

    private static bool CompareImages(Mat img1, Mat img2)
    {
        try
        {
            // Convert the images to grayscale
            CvInvoke.CvtColor(img1, img1, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(img2, img2, ColorConversion.Bgr2Gray);

            // Compute the mean of img1 and img2
            var mu1 = CvInvoke.Mean(img1);
            var mu2 = CvInvoke.Mean(img2);

            // Compute the standard deviation of img1 and img2
            var std1 = new MCvScalar();
            var std2 = new MCvScalar();
            CvInvoke.MeanStdDev(img1, ref mu1, ref std1);
            CvInvoke.MeanStdDev(img2, ref mu2, ref std2);

            // Compute the SSIM index
            var ssimIndex = ((2 * mu1.V0 * mu2.V0 + 1) * (2 * std1.V0 * std2.V0 + 1)) / ((mu1.V0 * mu1.V0 + mu2.V0 * mu2.V0 + 1) * (std1.V0 * std1.V0 + std2.V0 * std2.V0 + 1));

            // The SSIM index is a value between -1 and 1. Values closer to 1 mean more similarity.
            // adjust the threshold depending on your specific requirements.
            return ssimIndex > 0.75;
        }
        catch (AccessViolationException ex)
        {
            Console.WriteLine(ex.StackTrace);
        }

        throw new Exception();
    }
}
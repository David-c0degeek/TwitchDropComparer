using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace MissingRustTwitchDrops;

internal static class ImageComparator
{
    public static List<KeyValuePair<Mat, string>> Compare(IEnumerable<KeyValuePair<Mat, string>> images1, List<KeyValuePair<Mat, string>> images2)
    {
        return (
                from image1 in images1
                let matchFound = images2
                    .Any(image2 =>
                        CompareImagesUsingOrb(image1.Key, image2.Key))
                where !matchFound
                select image1)
            .ToList();
    }

    private static bool CompareImagesUsingOrb(Mat img1, Mat img2)
    {
        var orbDetector = new ORB();
        var keypoints1 = new VectorOfKeyPoint();
        var keypoints2 = new VectorOfKeyPoint();
        var descriptors1 = new Mat();
        var descriptors2 = new Mat();

        orbDetector.DetectAndCompute(img1, null, keypoints1, descriptors1, false);
        orbDetector.DetectAndCompute(img2, null, keypoints2, descriptors2, false);

        // Skip matching if either descriptor is empty
        if (descriptors1.IsEmpty || descriptors2.IsEmpty)
        {
            return false;
        }
        
        var matcher = new BFMatcher(DistanceType.Hamming);
        var matches = new VectorOfVectorOfDMatch();
        
        matcher.KnnMatch(descriptors1, descriptors2, matches, 2);

        // Apply ratio test
        var goodMatches = new List<MDMatch>();
        for (var i = 0; i < matches.Size; i++)
        {
            if (matches[i][0].Distance < 0.75 * matches[i][1].Distance)
            {
                goodMatches.Add(matches[i][0]);
            }
        }

        // The images are considered similar if we have a substantial number of good matches
        // Adjust the threshold depending on your specific requirements.
        return goodMatches.Count > 90;
    }

    
    private static bool CompareImagesUsingSsim(Mat img1, Mat img2)
    {
        try
        {
            // Convert the images to grayscale
            if (img1.NumberOfChannels > 1)
            {
                CvInvoke.CvtColor(img1, img1, ColorConversion.Bgr2Gray);
            }

            if (img2.NumberOfChannels > 1)
            {
                CvInvoke.CvtColor(img2, img2, ColorConversion.Bgr2Gray);
            }

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
        catch (Exception ex)
        {
            Console.WriteLine(ex.StackTrace);
        }

        throw new Exception();
    }
}
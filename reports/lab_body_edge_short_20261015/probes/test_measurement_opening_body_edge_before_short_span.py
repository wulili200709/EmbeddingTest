import json
import unittest

import numpy as np

from algorithms.measurement_opening_gap import OpeningGapRegion
from algorithms.measurement_opening_body_edge import measure_opening_body_edges


def sample(slope=.08, face_slope=None, bump=False, backing=False):
    yy,xx=np.mgrid[:300,:260]
    dx=xx-130
    upper=90+slope*dx
    lower=190+slope*dx
    if bump:
        lower=lower-8*(np.abs(dx)<12)
    image=np.full(xx.shape,130.,dtype=float)
    # Dark lips, whose external contour is independent of the bright face.
    image[yy<upper]=65
    image[yy>=lower]=65
    face_slope=slope if face_slope is None else face_slope
    image[yy<78+face_slope*dx]=230
    image[yy>=202+face_slope*dx]=230
    if backing:
        # Stronger white backing lies behind the lower lip in the search band.
        image[(yy>=202+face_slope*dx)&(yy<222+slope*dx)]=145
        image[yy>=222+slope*dx]=255
    return np.rint(image).astype(np.uint8)


REGION=OpeningGapRegion('synthetic',130,(50,115),(165,240),49)


class BodyContourTests(unittest.TestCase):
    def test_nonzero_tilt_tracks_body_normal_distance(self):
        result=measure_opening_body_edges(sample(.12),REGION,blur_ksize=3)
        self.assertNotEqual(result['status'],'failed')
        self.assertAlmostEqual(result['upper']['coefficients'][0],.12,delta=.015)
        self.assertAlmostEqual(result['lower']['coefficients'][0],.12,delta=.015)
        # A first-departure contour is near the pixel transition, with finite PSF.
        self.assertAlmostEqual(result['distance_px'],100/np.hypot(.12,1),delta=3)
        json.dumps(result,allow_nan=False)

    def test_internal_bright_face_slope_does_not_set_outer_line(self):
        result=measure_opening_body_edges(sample(0.,.09),REGION,blur_ksize=3)
        self.assertNotEqual(result['status'],'failed')
        self.assertAlmostEqual(result['upper']['coefficients'][0],0.,delta=.015)
        self.assertAlmostEqual(result['lower']['coefficients'][0],0.,delta=.015)
        self.assertLess(result['angle_difference_deg'],.5)

    def test_middle_protrusion_is_preserved_without_pulling_line(self):
        result=measure_opening_body_edges(sample(.06,bump=True),REGION,blur_ksize=3)
        self.assertAlmostEqual(result['lower']['coefficients'][0],.06,delta=.015)
        self.assertNotEqual(result['status'],'ok')
        self.assertGreater(result['lower']['middle_max_deviation_px'],5)
        self.assertGreater(result['lower']['middle_deviation_max_run'],4)
        self.assertTrue(any(result['lower']['middle_deviation_mask']))
        self.assertFalse(any(a and abs(p[0]-130)<40 for a,p in
                             zip(result['lower']['anchor_mask'],result['lower']['points'])))

    def test_no_contrast_rejects_with_finite_diagnostic_points(self):
        result=measure_opening_body_edges(np.full((300,260),130,np.uint8),REGION)
        self.assertEqual(result['status'],'failed')
        self.assertIsNone(result['distance_px'])
        self.assertFalse(any(result['upper']['valid_mask']))
        self.assertTrue(np.isfinite(result['upper']['points']).all())
        json.dumps(result,allow_nan=False)

    def test_stronger_backing_does_not_steal_inner_lip(self):
        result=measure_opening_body_edges(sample(.05,backing=True),REGION,blur_ksize=3)
        self.assertNotEqual(result['status'],'failed')
        a,b=result['lower']['coefficients']
        self.assertAlmostEqual(a*130+b,190,delta=3)
        self.assertGreater(len(result['lower']['seed_branches']),1)

    def test_real_step_is_not_silently_reported_as_straight(self):
        image=sample(0.)
        image[:,131:]=np.roll(image[:,131:],8,axis=0)
        result=measure_opening_body_edges(image,REGION,blur_ksize=3)
        self.assertNotEqual(result['status'],'ok')
        self.assertGreater(result['upper']['middle_max_deviation_px'],2.5)

    def test_curved_gap_background_cannot_create_floating_contour(self):
        image=sample(0.)
        # A gradual shadow flattens before the real lip. Linear extrapolation
        # of the shadow creates a residual crossing in an almost flat region.
        profile=np.interp(np.arange(300),[130,160,170,182,193,194,201,202,299],
                          [130,155,140,120,120,65,65,230,230])
        image[130:]=np.rint(profile[130:,None]).astype(np.uint8)
        result=measure_opening_body_edges(image,REGION)
        lower=result['lower']
        self.assertTrue(any(r['onset_rejections'] for r in lower['background_diagnostics']))
        # A usable result must be near the true y=194 contour. Rejection is
        # preferable to publishing a phantom edge in the flat shadow.
        if result['status']!='failed':
            a,b=lower['coefficients']
            self.assertGreater(a*130+b,189)
        json.dumps(result,allow_nan=False)

    def test_partial_inner_lip_is_not_replaced_by_full_backing(self):
        image=sample(0.,backing=True)
        image[190:222,:]=130
        # Only 45 columns expose the inner material; the backing spans all.
        image[190:202,108:153]=65
        image[202:222,108:153]=145
        result=measure_opening_body_edges(image,REGION,blur_ksize=3)
        self.assertEqual(result['status'],'failed')
        self.assertIsNone(result['distance_px'])


if __name__=='__main__':unittest.main()

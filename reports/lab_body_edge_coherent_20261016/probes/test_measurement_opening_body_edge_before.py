import json
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from algorithms.measurement_opening_gap import OpeningGapRegion
from algorithms.measurement_opening_body_edge import measure_opening_body_edges
from algorithms import measurement_opening_body_edge as body_module


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
    def test_blurred_step_localizes_edge_instead_of_diffusion_tail(self):
        # The physical sample transitions are exactly between rows 89/90 and
        # 189/190. Changing contrast and optical blur changes the low-percent
        # onset, but must not shift the reported edge into that onset tail.
        for background,lip in [(130,65),(180,75),(105,75)]:
            for optical_blur in (1,5,9):
                with self.subTest(background=background,lip=lip,blur=optical_blur):
                    image=np.full((300,260),background,np.float32)
                    image[:90]=lip
                    image[190:]=lip
                    image[:78]=230
                    image[202:]=230
                    if optical_blur>1:
                        image=cv2.GaussianBlur(image,(1,optical_blur),0)
                    result=measure_opening_body_edges(image,REGION,fit_width=101)
                    self.assertNotEqual(result['status'],'failed')
                    for side,target in [('upper',89.5),('lower',189.5)]:
                        edge=result[side]
                        a,b=edge['coefficients']
                        self.assertAlmostEqual(a*130+b,target,delta=.3)
                        self.assertEqual(edge['point_definition'],
                                         'same_local_transition_subpixel_gradient_peak')
                        valid=np.array(edge['valid_mask'])
                        onset=np.array(edge['onset_points'])[valid,1]
                        measured=np.array(edge['points'])[valid,1]
                        self.assertGreater(float(np.median(np.abs(onset-measured))),.8)
                    self.assertAlmostEqual(result['distance_px'],100.,delta=.4)

    def test_same_polarity_later_bright_face_does_not_replace_first_event(self):
        # Two rising transitions with a resolved gradient valley; the second
        # is stronger but belongs to an internal bright face.
        scan=np.arange(20,40)
        response=np.array([0,1,3,6,9,6,3,2,4,10,18,12,5,1,0,0,0,0,0],float)
        profile=np.zeros(80)
        profile[20:40]=np.r_[0,np.cumsum(response)]
        peak,evidence=body_module._transition_peak(profile,scan,3,1,.5,5.)
        self.assertAlmostEqual(peak,24.5,delta=.01)
        self.assertLess(peak,29)
        self.assertEqual(evidence['status'],'localized')

    def test_branch_extrapolation_into_gap_does_not_override_observed_contour(self):
        original=body_module._candidate_points
        def add_unsupported_extrapolation(*args,**kwargs):
            candidate=original(*args,**kwargs)
            if not args[4]:  # lower side
                candidate['branches'].append({'slope':-.32,'center_y':175.,
                    'support':45,'coverage':.46,'rmse_px':1.,'eligible':False})
            return candidate
        with patch.object(body_module,'_candidate_points',side_effect=add_unsupported_extrapolation):
            result=measure_opening_body_edges(sample(0.),REGION,fit_width=101,blur_ksize=3)
        self.assertNotEqual(result['status'],'failed')
        edge=result['lower']
        self.assertEqual(edge['unresolved_inner_seed_branches'],[])
        self.assertEqual(edge['inner_seed_branch_evidence'][-1]['openingward_support'],0)

    def test_conflicting_side_contours_reject_even_with_inliers_on_both_sides(self):
        yy,xx=np.mgrid[:300,:260]
        lower=np.full(xx.shape,190)
        # Each fitting side contains both the main contour and a deep recess.
        # A robust line retains enough inliers on both sides, but it cannot
        # represent these incompatible observations as one measured body edge.
        lower[((xx>=88)&(xx<96))|((xx>=165)&(xx<173))]+=15
        image=np.full(xx.shape,130,np.uint8)
        image[yy<90]=65
        image[yy<78]=230
        image[yy>=lower]=65
        image[yy>=lower+12]=230
        result=measure_opening_body_edges(image,REGION,fit_width=101,blur_ksize=3)
        edge=result['lower']
        self.assertNotIn('insufficient_two_sided_inlier_support',edge['quality_flags'])
        self.assertEqual(edge['inlier_tolerance_px'],2.5)
        self.assertEqual(edge['anchor_rmse_rejection_threshold_px'],5.)
        self.assertGreater(edge['anchor_rmse_px'],5.)
        self.assertIn('side_anchors_do_not_define_single_body_line',edge['quality_flags'])
        self.assertEqual(result['status'],'failed')
        self.assertIsNone(result['distance_px'])
        self.assertIsNotNone(result['diagnostic_distance_px'])

    def test_short_span_samples_and_fits_actual_requested_columns(self):
        for width in (101, 81):
            with self.subTest(width=width):
                result=measure_opening_body_edges(sample(.12),REGION,
                                                  fit_width=width,blur_ksize=3)
                self.assertNotEqual(result['status'],'failed')
                expected_span=[130-width//2,130+width//2]
                scale=(width-1)/140
                self.assertEqual(result['fit_width'],width)
                self.assertEqual(result['span'],expected_span)
                self.assertEqual(result['anchor_abs_dx_range_px'],[40*scale,65*scale])
                for side in ('upper','lower'):
                    edge=result[side]
                    xs=np.array(edge['points'])[:,0]
                    expected=(np.abs(xs-130)>=40*scale)&(np.abs(xs-130)<=65*scale)
                    self.assertEqual(len(xs),width)
                    self.assertEqual(xs.tolist(),list(range(expected_span[0],expected_span[1]+1)))
                    self.assertEqual([p[0] for p in edge['segment']],expected_span)
                    self.assertEqual(edge['anchor_zone_mask'],expected.tolist())
                    self.assertEqual(edge['anchor_x_ranges'],result['anchor_x_ranges'])
                    self.assertAlmostEqual(edge['coefficients'][0],.12,delta=.015)
                json.dumps(result,allow_nan=False)

    def test_short_span_seed_fallback_uses_scaled_side_sections(self):
        original=body_module._candidate_points
        def require_two_side_fallback(*args,**kwargs):
            candidate=original(*args,**kwargs)
            # A conservative coarse branch may fail eligibility even though
            # real candidates support both of the requested narrow sections.
            for branch in candidate['branches']:
                branch['eligible']=False
            return candidate
        with patch.object(body_module,'_candidate_points',side_effect=require_two_side_fallback):
            result=measure_opening_body_edges(sample(.06),REGION,fit_width=81,blur_ksize=3)
        self.assertNotEqual(result['status'],'failed')
        for side in ('upper','lower'):
            self.assertTrue(result[side]['seed_from_two_sided_candidate_support'])
            self.assertAlmostEqual(result[side]['coefficients'][0],.06,delta=.015)

    def test_outside_step_and_tilt_do_not_set_short_span_line(self):
        for width in (101,81):
            with self.subTest(width=width):
                original=sample(.06)
                changed=original.copy()
                # Beyond the three-pixel smoothing halo, replace the outer
                # portions by a large step and a different slope.
                for x in range(changed.shape[1]):
                    if abs(x-130)>width//2+3:
                        changed[:,x]=np.roll(original[:,x],18+int(.3*(x-130)),axis=0)
                before=measure_opening_body_edges(original,REGION,fit_width=width,blur_ksize=3)
                after=measure_opening_body_edges(changed,REGION,fit_width=width,blur_ksize=3)
                self.assertNotEqual(after['status'],'failed')
                self.assertAlmostEqual(before['distance_px'],after['distance_px'],places=10)
                for side in ('upper','lower'):
                    np.testing.assert_allclose(before[side]['points'],after[side]['points'],atol=1e-10)
                    np.testing.assert_allclose(before[side]['coefficients'],after[side]['coefficients'],atol=1e-10)

    def test_short_span_preserves_tilt_and_central_protrusion(self):
        for width in (101,81):
            with self.subTest(width=width):
                flat=measure_opening_body_edges(sample(.09),REGION,fit_width=width,blur_ksize=3)
                bumped=measure_opening_body_edges(sample(.09,bump=True),REGION,
                                                  fit_width=width,blur_ksize=3)
                self.assertNotEqual(bumped['status'],'failed')
                self.assertAlmostEqual(bumped['lower']['coefficients'][0],.09,delta=.015)
                self.assertAlmostEqual(flat['distance_px'],bumped['distance_px'],delta=.2)
                self.assertGreater(bumped['lower']['middle_max_deviation_px'],5)
                self.assertGreater(bumped['lower']['middle_deviation_max_run'],4)
                self.assertIn('middle_contour_departs_from_body_line',bumped['lower']['quality_flags'])

    def test_nonzero_tilt_tracks_body_normal_distance(self):
        result=measure_opening_body_edges(sample(.12),REGION,blur_ksize=3)
        self.assertNotEqual(result['status'],'failed')
        self.assertAlmostEqual(result['upper']['coefficients'][0],.12,delta=.015)
        self.assertAlmostEqual(result['lower']['coefficients'][0],.12,delta=.015)
        self.assertAlmostEqual(result['distance_px'],100/np.hypot(.12,1),delta=.6)
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

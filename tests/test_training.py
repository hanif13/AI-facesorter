import math
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src.models.arcface_head import ArcFaceHead
from src.models.mobilefacenet import MobileFaceNet

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_mobilefacenet_both_sizes_and_init_proof(self):
        for size in [96,112]:
            model=MobileFaceNet(size,128)
            proof=model.assert_random_init()
            self.assertEqual(proof["origin"],"random_init")
            self.assertEqual(model(torch.randn(2,3,size,size)).shape,(2,128))
            with self.assertRaises(AssertionError): model.assert_random_init()  # BN state changed after training-mode forward.

    def test_zero_margin_matches_normalized_softmax(self):
        head=ArcFaceHead(4,8,s=16)
        x=torch.randn(3,8); y=torch.tensor([0,1,3])
        logits,cos=head(x,y,margin=0)
        expected=F.linear(F.normalize(x,dim=1),F.normalize(head.weight,dim=1))
        torch.testing.assert_close(logits,16*expected)

    def test_target_margin_formula_and_non_target_logits(self):
        head=ArcFaceHead(2,2,s=1,m=.5)
        with torch.no_grad(): head.weight.copy_(torch.eye(2))
        x=torch.tensor([[math.cos(.8),math.sin(.8)]],requires_grad=True)
        logits,cos=head(x,torch.tensor([0]))
        self.assertAlmostEqual(float(logits[0,0].detach()),math.cos(1.3),places=5)
        torch.testing.assert_close(logits[0,1],cos[0,1])
        logits.sum().backward(); self.assertTrue(torch.isfinite(x.grad).all())

    def test_margin_does_not_wrap_past_pi(self):
        head=ArcFaceHead(2,2,s=1,m=.5)
        with torch.no_grad(): head.weight.copy_(torch.eye(2))
        logits,cos=head(torch.tensor([[-1.,.01]]),torch.tensor([0]))
        expected=cos[0,0]-math.sin(math.pi-.5)*.5
        torch.testing.assert_close(logits[0,0],expected)
        self.assertLess(float(logits[0,0].detach()),float(cos[0,0].detach()))

    def test_verification_threshold_ignores_held_out_labels(self):
        from src.eval_lfw import verification
        # Synthetic scores for a protocol test, not an LFW result.
        same=np.array([True,False]*10)
        vectors=[]
        for value in same:
            vectors.extend([[1.,0.],[1.,0.] if value else [0.,1.]])
        with tempfile.TemporaryDirectory() as temp:
            first=verification(np.array(vectors),same,Path(temp)/"first")
            changed=same.copy(); changed[:2]=~changed[:2]
            second=verification(np.array(vectors),changed,Path(temp)/"changed")
        self.assertEqual(first["fold_thresholds"][0],second["fold_thresholds"][0])

if __name__=="__main__": unittest.main()

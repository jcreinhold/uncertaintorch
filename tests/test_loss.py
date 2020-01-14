#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_loss

test the uncertaintorch loss functions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jan 14, 2020
"""

import unittest

import torch

from uncertaintorch.learn import *


class TestLoss(unittest.TestCase):

    def setUp(self):
        pass

    def test_mseonlyloss_nomask(self):
        fn = MSEOnlyLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,1,2,2,2))
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_mseonlyloss_mask(self):
        fn = MSEOnlyLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,2,2,2,2))
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_gaussiandiagloss_nomask(self):
        fn = GaussianDiagLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,1,2,2,2))
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_gaussiandiagloss_mask(self):
        fn = GaussianDiagLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,2,2,2,2))
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_l1onlyloss_nomask(self):
        fn = L1OnlyLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,1,2,2,2))
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_l1onlyloss_mask(self):
        fn = L1OnlyLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,2,2,2,2))
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_laplaciandiagloss_nomask(self):
        fn = LaplacianDiagLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,1,2,2,2))
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_laplaciandiagloss_mask(self):
        fn = LaplacianDiagLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,2,2,2,2))
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_focalloss_nomask(self):
        fn = FocalLoss()
        x, y = torch.zeros((2,2,2,2,2)), torch.zeros((2,2,2,2),dtype=torch.long)
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_focalloss_mask(self):
        fn = FocalLoss()
        x, y = torch.zeros((2,2,2,2,2)), torch.zeros((2,2,2,2),dtype=torch.long)
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_extendedcrossentropy_nomask(self):
        fn = ExtendedCrossEntropy()
        x, y = (torch.zeros((2,2,2,2,2)), torch.zeros((2,2,2,2,2))), torch.zeros((2,2,2,2),dtype=torch.long)
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def test_extendedcrossentropy_mask(self):
        fn = ExtendedCrossEntropy()
        x, y = (torch.zeros((2,2,2,2,2)), torch.zeros((2,2,2,2,2))), torch.zeros((2,2,2,2),dtype=torch.long)
        loss = fn(x, y)
        self.assertEqual(loss.item(), 0)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()


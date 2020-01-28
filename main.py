#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:02:16 2020

@author: ishwark
"""

from trainer import UnsupervisedTrainer

if __name__ == "__main__":
    try:
        trainer = UnsupervisedTrainer('MNIST')
        trainer.train_loop()
    except KeyboardInterrupt:
        if trainer:
            print("Keyboard Interrupt, saving model and quitting\n")
            trainer.save_model("final_epoch.model")

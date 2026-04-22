"""
Unit tests for the design pattern classes
Run with: pytest test_patterns.py -v
"""

import pytest
import torch


class TestGPTConfigBuilder:

  def test_valid_config_builds(self):
    from gpt_config_builder import GPTConfigBuilder
    cfg = (GPTConfigBuilder()
                .set_vocab_size(8000)
                .set_model_dim(256)
                .set_heads(4)
                .set_layers(4)
                .build())
    assert cfg.vocab_size == 8000
    assert cfg.d_model == 256

  def test_incompatible_dim_and_heads_raises(self):
    from gpt_config_builder import GPTConfigBuilder
    with pytest.raises(ValueError):
      GPTConfigBuilder().set_model_dim(256).set_heads(5).build()

  def test_invalid_dropout_raises(self):
    from gpt_config_builder import GPTConfigBuilder
    with pytest.raises(ValueError):
      GPTConfigBuilder().set_dropout(2.0)


class TestTokenizerSingleton:

  def setup_method(self):
    from tokenizer_singleton import TokenizerSingleton
    TokenizerSingleton.reset()

  def test_returns_same_instance_twice(self):
    from tokenizer_singleton import TokenizerSingleton
    from unittest.mock import patch, MagicMock
    with patch("tokenizer_singleton.Tokenizer.from_file", return_value=MagicMock()):
      t1 = TokenizerSingleton.get_instance("dummy.json")
      t2 = TokenizerSingleton.get_instance("dummy.json")
    assert t1 is t2

  def test_file_only_loaded_once(self):
    from tokenizer_singleton import TokenizerSingleton
    from unittest.mock import patch, MagicMock
    with patch("tokenizer_singleton.Tokenizer.from_file", return_value=MagicMock()) as mock_load:
      TokenizerSingleton.get_instance("dummy.json")
      TokenizerSingleton.get_instance("dummy.json")
    assert mock_load.call_count == 1


class TestModelFactory:

  def test_standard_model_has_correct_config(self):
    from model_factory import ModelFactory
    model = ModelFactory().create_model("standard")
    assert model.cfg.d_model == 256
    assert model.cfg.n_layers == 4

  def test_unknown_model_type_raises(self):
    from model_factory import ModelFactory
    with pytest.raises(ValueError):
      ModelFactory().create_model("unknown")


class TestDecodingStrategies:

  def _logits(self, vocab_size=10, best=3):
    logits = torch.zeros(1, vocab_size)
    logits[0, best] = 10.0
    return logits

  def test_greedy_always_picks_highest(self):
    from decoding_strategy import GreedyStrategy
    assert GreedyStrategy().select_token(self._logits(best=3)).item() == 3

  def test_greedy_output_is_shape_1x1(self):
    from decoding_strategy import GreedyStrategy
    assert GreedyStrategy().select_token(self._logits()).shape == (1, 1)

  def test_topk_output_is_shape_1x1(self):
    from decoding_strategy import TopKStrategy
    assert TopKStrategy(k=3).select_token(self._logits()).shape == (1, 1)

  def test_nucleus_output_is_shape_1x1(self):
    from decoding_strategy import NucleusStrategy
    assert NucleusStrategy(p=0.9).select_token(self._logits()).shape == (1, 1)




class TestObserverPattern:

  def _event(self, val_loss=None):
    from training_observer import TrainingEvent
    return TrainingEvent(step=1, epoch=1, train_loss=2.0, val_loss=val_loss)

  def test_observer_receives_event(self):
    from training_observer import TrainingSubject, TrainingObserver
    received = []
    class Spy(TrainingObserver):
      def on_event(self, event): received.append(event)
    subject = TrainingSubject()
    subject.attach(Spy())
    subject.notify(self._event())
    assert len(received) == 1

  def test_early_stopping_triggers_after_patience(self):
    from training_observer import EarlyStoppingObserver
    stopper = EarlyStoppingObserver(patience=2, min_delta=0.01)
    event = self._event(val_loss=5.0)
    stopper.on_event(event)
    stopper.on_event(event)
    with pytest.raises(StopIteration):
      stopper.on_event(event)
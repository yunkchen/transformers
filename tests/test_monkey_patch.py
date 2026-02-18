# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import threading
import unittest

from transformers import is_torch_available
from transformers.monkey_patch import (
    apply_monkey_patches,
    clear_monkey_patch_mapping,
    get_monkey_patch_mapping,
    patch_output_recorders,
    register_monkey_patch_mapping,
    unregister_monkey_patch_mapping,
)
from transformers.testing_utils import require_torch
from transformers.utils.output_capturing import OutputRecorder


if is_torch_available():
    import torch.nn as nn


@require_torch
class MonkeyPatchTest(unittest.TestCase):
    def setUp(self):
        """Clear any existing patches before each test."""
        clear_monkey_patch_mapping()

    def tearDown(self):
        """Clean up patches after each test."""
        clear_monkey_patch_mapping()

    def test_register_monkey_patch_mapping(self):
        """Test basic registration of monkey patches."""

        class CustomModule(nn.Module):
            pass

        # Register a patch
        register_monkey_patch_mapping(mapping={"TestModule": CustomModule})

        # Verify it was registered
        mapping = get_monkey_patch_mapping()
        self.assertIn("TestModule", mapping)
        self.assertEqual(mapping["TestModule"], CustomModule)

    def test_register_multiple_patches(self):
        """Test registering multiple patches at once."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        # Register multiple patches
        register_monkey_patch_mapping(mapping={"TestModule1": CustomModule1, "TestModule2": CustomModule2})

        # Verify both were registered
        mapping = get_monkey_patch_mapping()
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping["TestModule1"], CustomModule1)
        self.assertEqual(mapping["TestModule2"], CustomModule2)

    def test_register_duplicate_without_overwrite_raises_error(self):
        """Test that registering a duplicate class without overwrite raises an error."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        # Register initial patch
        register_monkey_patch_mapping(mapping={"TestModule": CustomModule1})

        # Try to register same class name without overwrite
        with self.assertRaises(ValueError) as context:
            register_monkey_patch_mapping(mapping={"TestModule": CustomModule2})

        self.assertIn("already has a patch mapping registered", str(context.exception))
        self.assertIn("overwrite=True", str(context.exception))

    def test_register_duplicate_with_overwrite(self):
        """Test that registering a duplicate class with overwrite=True works."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        # Register initial patch
        register_monkey_patch_mapping(mapping={"TestModule": CustomModule1})

        # Overwrite with new patch
        register_monkey_patch_mapping(mapping={"TestModule": CustomModule2}, overwrite=True)

        # Verify the new patch is registered
        mapping = get_monkey_patch_mapping()
        self.assertEqual(mapping["TestModule"], CustomModule2)

    def test_register_non_class_raises_error(self):
        """Test that registering a non-class raises TypeError."""

        # Try to register an instance instead of a class
        with self.assertRaises(TypeError) as context:
            register_monkey_patch_mapping(mapping={"TestModule": nn.Module()})

        self.assertIn("must be a class", str(context.exception))

    def test_register_non_nn_module_raises_error(self):
        """Test that registering a non-nn.Module class raises TypeError."""

        class NotAModule:
            pass

        # Try to register a non-nn.Module class
        with self.assertRaises(TypeError) as context:
            register_monkey_patch_mapping(mapping={"TestModule": NotAModule})

        self.assertIn("must be a subclass of nn.Module", str(context.exception))

    def test_unregister_monkey_patch_mapping(self):
        """Test unregistering monkey patches."""

        class CustomModule(nn.Module):
            pass

        # Register and then unregister
        register_monkey_patch_mapping(mapping={"TestModule": CustomModule})
        unregister_monkey_patch_mapping(["TestModule"])

        # Verify it was unregistered
        mapping = get_monkey_patch_mapping()
        self.assertNotIn("TestModule", mapping)

    def test_unregister_nonexistent_class(self):
        """Test unregistering a class that doesn't exist (should not raise error)."""
        # This should not raise an error
        unregister_monkey_patch_mapping(["NonexistentModule"])

    def test_unregister_multiple_classes(self):
        """Test unregistering multiple classes at once."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        class CustomModule3(nn.Module):
            pass

        # Register three patches
        register_monkey_patch_mapping(
            mapping={"TestModule1": CustomModule1, "TestModule2": CustomModule2, "TestModule3": CustomModule3}
        )

        # Unregister two
        unregister_monkey_patch_mapping(["TestModule1", "TestModule2"])

        # Verify only one remains
        mapping = get_monkey_patch_mapping()
        self.assertEqual(len(mapping), 1)
        self.assertIn("TestModule3", mapping)

    def test_clear_monkey_patch_mapping(self):
        """Test clearing all monkey patches."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        # Register multiple patches
        register_monkey_patch_mapping(mapping={"TestModule1": CustomModule1, "TestModule2": CustomModule2})

        # Clear all patches
        clear_monkey_patch_mapping()

        # Verify all were cleared
        mapping = get_monkey_patch_mapping()
        self.assertEqual(len(mapping), 0)

    def test_get_monkey_patch_mapping_returns_copy(self):
        """Test that get_monkey_patch_mapping returns a copy, not the original."""

        class CustomModule(nn.Module):
            pass

        register_monkey_patch_mapping(mapping={"TestModule": CustomModule})

        # Get mapping and modify it
        mapping = get_monkey_patch_mapping()
        mapping["NewModule"] = CustomModule

        # Verify the internal cache was not modified
        internal_mapping = get_monkey_patch_mapping()
        self.assertNotIn("NewModule", internal_mapping)

    def test_apply_monkey_patches_context_manager(self):
        """Test that apply_monkey_patches context manager works correctly."""

        class CustomLinear(nn.Linear):
            pass

        # Create a dummy module in transformers namespace for testing
        import types

        test_module = types.ModuleType("transformers.test_module")
        test_module.Linear = nn.Linear
        sys.modules["transformers.test_module"] = test_module

        try:
            # Register patch
            register_monkey_patch_mapping(mapping={"Linear": CustomLinear})

            # Outside context, original class should be used
            self.assertEqual(test_module.Linear, nn.Linear)

            # Inside context, patched class should be used
            with apply_monkey_patches():
                self.assertEqual(test_module.Linear, CustomLinear)

            # Outside context again, original class should be restored
            self.assertEqual(test_module.Linear, nn.Linear)

        finally:
            # Clean up the test module
            del sys.modules["transformers.test_module"]

    def test_apply_monkey_patches_with_empty_mapping(self):
        """Test that apply_monkey_patches with empty mapping does nothing."""
        # Clear all patches
        clear_monkey_patch_mapping()

        # This should work without errors
        with apply_monkey_patches():
            pass

    def test_apply_monkey_patches_multiple_modules(self):
        """Test patching the same class across multiple transformers modules."""

        class CustomLinear(nn.Linear):
            pass

        # Create multiple test modules
        import types

        test_module1 = types.ModuleType("transformers.test_module1")
        test_module1.Linear = nn.Linear
        sys.modules["transformers.test_module1"] = test_module1

        test_module2 = types.ModuleType("transformers.test_module2")
        test_module2.Linear = nn.Linear
        sys.modules["transformers.test_module2"] = test_module2

        try:
            register_monkey_patch_mapping(mapping={"Linear": CustomLinear})

            with apply_monkey_patches():
                # Both modules should be patched
                self.assertEqual(test_module1.Linear, CustomLinear)
                self.assertEqual(test_module2.Linear, CustomLinear)

            # Both should be restored
            self.assertEqual(test_module1.Linear, nn.Linear)
            self.assertEqual(test_module2.Linear, nn.Linear)

        finally:
            # Clean up
            del sys.modules["transformers.test_module1"]
            del sys.modules["transformers.test_module2"]

    def test_thread_safety_concurrent_registration(self):
        """Test that concurrent registrations are thread-safe."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        class CustomModule3(nn.Module):
            pass

        results = []
        errors = []

        def register_patches(class_name, module_class):
            try:
                register_monkey_patch_mapping(mapping={class_name: module_class})
                results.append(class_name)
            except Exception as e:
                errors.append((class_name, e))

        # Create multiple threads that register different patches
        threads = [
            threading.Thread(target=register_patches, args=("Module1", CustomModule1)),
            threading.Thread(target=register_patches, args=("Module2", CustomModule2)),
            threading.Thread(target=register_patches, args=("Module3", CustomModule3)),
        ]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors occurred during concurrent registration: {errors}")

        # Verify all patches were registered
        mapping = get_monkey_patch_mapping()
        self.assertEqual(len(mapping), 3)
        self.assertIn("Module1", mapping)
        self.assertIn("Module2", mapping)
        self.assertIn("Module3", mapping)

    def test_thread_safety_concurrent_access(self):
        """Test that concurrent reads and writes are thread-safe."""

        class CustomModule(nn.Module):
            pass

        results = []

        def read_mapping():
            for _ in range(100):
                mapping = get_monkey_patch_mapping()
                results.append(len(mapping))

        def write_mapping():
            for i in range(100):
                mapping = {f"Module{i}": CustomModule}
                register_monkey_patch_mapping(mapping=mapping)

        # Create threads for reading and writing
        read_thread = threading.Thread(target=read_mapping)
        write_thread = threading.Thread(target=write_mapping)

        read_thread.start()
        write_thread.start()

        read_thread.join()
        write_thread.join()

        # Test should complete without deadlocks or errors
        self.assertEqual(len(results), 100)

    def test_patch_output_recorders_with_output_recorder_instance(self):
        """Test patching output recorders that are OutputRecorder instances."""

        class OriginalModule(nn.Module):
            pass

        class ReplacementModule(nn.Module):
            pass

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                # Simulate _can_record_outputs with OutputRecorder
                self._can_record_outputs = {"output": OutputRecorder(OriginalModule)}

        model = TestModel()

        # Register patch
        register_monkey_patch_mapping(mapping={"OriginalModule": ReplacementModule})

        # Patch output recorders
        patch_output_recorders(model)

        # Verify the recorder's target_class was updated
        recorder = model._can_record_outputs["output"]
        self.assertEqual(recorder.target_class, ReplacementModule)

    def test_patch_output_recorders_with_class_type(self):
        """Test patching output recorders that are class types directly."""

        class OriginalModule(nn.Module):
            pass

        class ReplacementModule(nn.Module):
            pass

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                # Simulate _can_record_outputs with class type directly
                self._can_record_outputs = {"output": OriginalModule}

        model = TestModel()

        # Register patch
        register_monkey_patch_mapping(mapping={"OriginalModule": ReplacementModule})

        # Patch output recorders
        patch_output_recorders(model)

        # Verify the class was updated
        self.assertEqual(model._can_record_outputs["output"], ReplacementModule)

    def test_patch_output_recorders_with_no_patches(self):
        """Test that patch_output_recorders with no registered patches does nothing."""

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = TestModel()

        # Should not raise any errors
        patch_output_recorders(model)

    def test_patch_output_recorders_with_nested_modules(self):
        """Test patching output recorders in nested modules."""

        class OriginalModule(nn.Module):
            pass

        class ReplacementModule(nn.Module):
            pass

        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self._can_record_outputs = {"output": OutputRecorder(OriginalModule)}

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = SubModule()

        model = TestModel()

        # Register patch
        register_monkey_patch_mapping(mapping={"OriginalModule": ReplacementModule})

        # Patch output recorders
        patch_output_recorders(model)

        # Verify nested submodule's recorder was updated
        recorder = model.submodule._can_record_outputs["output"]
        self.assertEqual(recorder.target_class, ReplacementModule)

    def test_context_manager_with_module_without_name(self):
        """Test that context manager handles modules without __name__ attribute gracefully."""

        class CustomModule(nn.Module):
            pass

        # Create a module-like object without __name__
        import types

        test_module = types.ModuleType("transformers.test_no_name")
        # Simulate a module in sys.modules that might not have proper attributes
        sys.modules["transformers.test_no_name"] = test_module

        try:
            register_monkey_patch_mapping(mapping={"TestClass": CustomModule})

            # This should not raise an error
            with apply_monkey_patches():
                pass

        finally:
            del sys.modules["transformers.test_no_name"]

    def test_incremental_registration(self):
        """Test that patches can be added incrementally."""

        class CustomModule1(nn.Module):
            pass

        class CustomModule2(nn.Module):
            pass

        class CustomModule3(nn.Module):
            pass

        # Register patches incrementally
        register_monkey_patch_mapping(mapping={"Module1": CustomModule1})
        self.assertEqual(len(get_monkey_patch_mapping()), 1)

        register_monkey_patch_mapping(mapping={"Module2": CustomModule2})
        self.assertEqual(len(get_monkey_patch_mapping()), 2)

        register_monkey_patch_mapping(mapping={"Module3": CustomModule3})
        mapping = get_monkey_patch_mapping()
        self.assertEqual(len(mapping), 3)

        # Verify all are registered
        self.assertEqual(mapping["Module1"], CustomModule1)
        self.assertEqual(mapping["Module2"], CustomModule2)
        self.assertEqual(mapping["Module3"], CustomModule3)


if __name__ == "__main__":
    unittest.main()

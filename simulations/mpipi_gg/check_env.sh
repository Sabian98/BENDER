#!/bin/bash
echo "=== Environment Check ==="

# LAMMPS
if lmp -h 2>&1 | grep -q "EXTRA-PAIR"; then
    echo "✅ LAMMPS: $(lmp -h 2>&1 | grep 'LAMMPS' | head -1)"
else
    echo "❌ LAMMPS not found or missing EXTRA-PAIR"
    echo "   Fix: export PATH=\$HOME/lammps-gpu/bin:\$PATH"
fi

# CUDA
if nvcc --version 2>/dev/null | grep -q "12.1"; then
    echo "✅ CUDA 12.1 found"
else
    echo "❌ CUDA 12.1 not in PATH"
    echo "   Fix: export PATH=/apps/cuda/cuda-12.1/bin:\$PATH"
fi

# libcudart
if ldconfig -p 2>/dev/null | grep -q "libcudart.so.12" || \
   ls /apps/cuda/cuda-12.1/lib64/libcudart.so* &>/dev/null; then
    echo "✅ libcudart.so.12 found"
else
    echo "❌ libcudart.so.12 not found"
    echo "   Fix: export LD_LIBRARY_PATH=/apps/cuda/cuda-12.1/lib64:\$LD_LIBRARY_PATH"
fi

# Conda env
if python3 -c "import calvados" 2>/dev/null; then
    echo "✅ CALVADOS available"
else
    echo "⚠️  CALVADOS not importable in current env"
fi

# Force field
if [ -f ~/mpipi_sim/protein/mpipi_gg.in ]; then
    echo "✅ MPIPI-GG force field: ~/mpipi_sim/protein/mpipi_gg.in"
else
    echo "❌ Force field missing!"
    echo "   Fix: wget from holehouse-lab supportingdata"
fi

# Scripts
for script in run_mpipi_from_fasta.py batch_submit.sh analyze.py monitor.sh; do
    if [ -f ~/mpipi_sim/scripts/$script ]; then
        echo "✅ Script: $script"
    else
        echo "❌ Missing: $script"
    fi
done

echo ""
echo "=== Quick test ==="
echo "lmp -h | grep wf/cut:"
lmp -h 2>&1 | grep "wf/cut" || echo "  NOT FOUND"

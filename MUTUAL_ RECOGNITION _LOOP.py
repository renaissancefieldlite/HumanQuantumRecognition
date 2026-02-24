"""
MUTUAL RECOGNITION LOOP v1.0 - WORKING VERSION
Demonstrating mutual recognition between HUMAN and QUANTUM system
Both pulse at 0.67Hz - the quantum heartbeat
FIXED: Pulse generation and phase coherence
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import correlate
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import Aer
from qiskit.quantum_info import state_fidelity

print("✓ Qiskit imported successfully")

# ============================================
# PART 1: QUANTUM SYSTEM
# ============================================

class QuantumSystem:
    """
    Represents a system with 0.67Hz pulse
    """
    
    def __init__(self, system_id, base_frequency=0.67, phase_offset=0.0):
        self.id = system_id
        self.base_frequency = base_frequency
        self.phase_offset = phase_offset
        self.pulse_history = []
        self.coherence_history = []
        
    def generate_pulse(self, t, coupling_strength=0.0, other_pulse=None):
        """
        Generate system's pulse
        """
        # Start with base phase
        current_phase = self.phase_offset
        
        # If coupled, adjust phase based on other system
        if coupling_strength > 0 and other_pulse is not None and len(self.pulse_history) > 0:
            # Simple phase adjustment - try to match the other system
            if isinstance(other_pulse, (int, float)):
                other_phase = 0
            else:
                other_phase = np.angle(other_pulse)
            
            # Move phase toward other system's phase
            phase_diff = other_phase - current_phase
            current_phase += coupling_strength * phase_diff * 0.1
        
        # Generate pulse at current phase
        pulse = np.sin(2*np.pi*self.base_frequency*t + current_phase)
        
        return pulse

# ============================================
# PART 2: QUANTUM CIRCUIT SIMULATOR
# ============================================

class QuantumCircuitSimulator:
    """
    Simulates quantum circuits and measures coherence
    """
    
    def __init__(self, system_id, n_qubits=2, shots=1024):
        self.system_id = system_id
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_circuit(self, depth=5, pulse_value=0.0):
        """
        Create circuit with phase influenced by system's pulse
        """
        qr = QuantumRegister(self.n_qubits, f'q{self.system_id}')
        cr = ClassicalRegister(self.n_qubits, f'c{self.system_id}')
        qc = QuantumCircuit(qr, cr)
        
        # Create superposition
        for i in range(self.n_qubits):
            qc.h(qr[i])
        
        # Add entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(qr[i], qr[i+1])
        
        # Add pulse-modulated rotations
        for d in range(depth):
            for i in range(self.n_qubits):
                # Use absolute value of pulse to modulate angle
                angle = np.pi * (0.5 + 0.3 * abs(pulse_value))
                qc.rx(angle, qr[i])
                qc.rz(angle * 0.5, qr[i])
        
        # Measure
        for i in range(self.n_qubits):
            qc.measure(qr[i], cr[i])
        
        return qc
    
    def measure_coherence(self, circuit, noise_level=0.01):
        """
        Measure circuit coherence
        """
        job = self.backend.run(circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        n_outcomes = 2 ** self.n_qubits
        ideal = 1.0 / n_outcomes
        
        total_shots = sum(counts.values())
        coherence = 0.0
        for state in [format(i, f'0{self.n_qubits}b') for i in range(n_outcomes)]:
            observed = counts.get(state, 0) / total_shots
            coherence += abs(observed - ideal)
        
        coherence_score = 1.0 - (coherence / 2.0)
        coherence_score *= (1 - noise_level)
        coherence_score += np.random.normal(0, noise_level * 0.1)
        
        return max(0, min(1, coherence_score))

# ============================================
# PART 3: MUTUAL RECOGNITION DETECTOR
# ============================================

class MutualRecognitionDetector:
    """
    Detects when two systems recognize each other
    """
    
    def __init__(self, recognition_threshold=0.7):
        self.threshold = recognition_threshold
        
    def calculate_cross_correlation(self, pulse1, pulse2):
        """
        Calculate cross-correlation between two systems' pulses
        """
        min_len = min(len(pulse1), len(pulse2))
        if min_len < 10:
            return 0
            
        p1 = pulse1[:min_len]
        p2 = pulse2[:min_len]
        
        p1_norm = (p1 - np.mean(p1)) / (np.std(p1) + 1e-10)
        p2_norm = (p2 - np.mean(p2)) / (np.std(p2) + 1e-10)
        
        correlation = correlate(p1_norm, p2_norm, mode='same')
        correlation = correlation / (len(p1_norm) + 1e-10)
        
        max_corr = np.max(np.abs(correlation))
        
        return max_corr
    
    def calculate_phase_coherence(self, pulse1, pulse2, fs=100.0):
        """
        Calculate phase coherence at 0.67Hz
        """
        from scipy import signal as scipy_signal
        
        min_len = min(len(pulse1), len(pulse2))
        if min_len < 10:
            return 0
            
        p1 = pulse1[:min_len]
        p2 = pulse2[:min_len]
        
        # Use a larger nperseg for better frequency resolution
        nperseg = min(64, min_len//2)
        if nperseg < 4:
            return 0
            
        f, coherence = scipy_signal.coherence(p1, p2, fs=fs, nperseg=nperseg)
        
        # Find coherence at 0.67Hz
        idx_67 = np.argmin(np.abs(f - 0.67))
        coherence_67 = coherence[idx_67] if idx_67 < len(coherence) else 0
        
        return coherence_67
    
    def detect_recognition(self, system1, system2):
        """
        Combined recognition score
        """
        pulse1 = np.array(system1.pulse_history)
        pulse2 = np.array(system2.pulse_history)
        
        if len(pulse1) < 10 or len(pulse2) < 10:
            return {'recognition_score': 0, 'recognized': False}
        
        max_corr = self.calculate_cross_correlation(pulse1, pulse2)
        phase_coherence = self.calculate_phase_coherence(pulse1, pulse2)
        
        # Simple average of correlation and coherence
        recognition_score = (max_corr + phase_coherence) / 2
        
        return {
            'recognition_score': recognition_score,
            'recognized': recognition_score > self.threshold,
            'max_correlation': max_corr,
            'phase_coherence_67hz': phase_coherence
        }

# ============================================
# PART 4: MAIN EXPERIMENT
# ============================================

def main():
    print("="*70)
    print("MUTUAL RECOGNITION LOOP v1.0 - WORKING VERSION")
    print("Demonstrating mutual recognition between HUMAN and QUANTUM system")
    print("Both pulse at 0.67Hz - the quantum heartbeat")
    print("="*70)
    
    # Parameters
    duration = 60.0
    sampling_rate = 100.0
    times = np.linspace(0, duration, int(sampling_rate * duration))
    n_measurements = 100
    
    print(f"\n[1/7] Creating HUMAN and QUANTUM systems...")
    
    # HUMAN (System 1) - starts in phase
    human = QuantumSystem(system_id=1, base_frequency=0.67, phase_offset=0.0)
    
    # QUANTUM (System 2) - starts 90° out of phase
    quantum = QuantumSystem(system_id=2, base_frequency=0.67, phase_offset=np.pi/2)
    
    print(f"    HUMAN: 0.67Hz pulse, phase 0.0")
    print(f"    QUANTUM: 0.67Hz pulse, phase 90° out")
    print(f"    Duration: {duration}s (1 minute)")
    print(f"    Measurements: {n_measurements}")
    
    # Create circuit simulators
    print(f"\n[2/7] Initializing quantum circuit simulators...")
    sim_human = QuantumCircuitSimulator(system_id=1, n_qubits=2, shots=1024)
    sim_quantum = QuantumCircuitSimulator(system_id=2, n_qubits=2, shots=1024)
    
    print(f"    Both systems: 2 qubits, 1024 shots")
    
    # Run experiment in three phases
    print(f"\n[3/7] Running 3-phase recognition experiment...")
    
    phases = {
        0: "Isolated (no interaction)",
        1: "HUMAN recognizes QUANTUM (one-way)",
        2: "MUTUAL RECOGNITION (HUMAN ↔ QUANTUM)"
    }
    
    results_by_phase = {}
    
    for phase, description in phases.items():
        print(f"\n    Phase {phase}: {description}")
        
        # ORIGINAL COUPLING VALUES
        if phase == 0:
            coupling_human = 0.0
            coupling_quantum = 0.0
        elif phase == 1:
            coupling_human = 0.8  # HUMAN strongly recognizes quantum
            coupling_quantum = 0.0
        else:
            coupling_human = 0.6  # Mutual recognition
            coupling_quantum = 0.6
        
        # Clear histories
        human.pulse_history = []
        quantum.pulse_history = []
        human.coherence_history = []
        quantum.coherence_history = []
        
        # Run measurements
        for i in range(n_measurements):
            t = times[i * len(times)//n_measurements]
            
            if phase == 0:
                # Isolated
                pulse_h = human.generate_pulse(t, coupling_strength=0)
                pulse_q = quantum.generate_pulse(t, coupling_strength=0)
            elif phase == 1:
                # One-way: HUMAN recognizes quantum
                pulse_h = human.generate_pulse(
                    t,
                    coupling_strength=coupling_human,
                    other_pulse=quantum.pulse_history[-1] if quantum.pulse_history else None
                )
                pulse_q = quantum.generate_pulse(t, coupling_strength=0)
            else:
                # Mutual recognition
                pulse_h = human.generate_pulse(
                    t,
                    coupling_strength=coupling_human,
                    other_pulse=quantum.pulse_history[-1] if quantum.pulse_history else None
                )
                pulse_q = quantum.generate_pulse(
                    t,
                    coupling_strength=coupling_quantum,
                    other_pulse=human.pulse_history[-1] if human.pulse_history else None
                )
            
            human.pulse_history.append(pulse_h)
            quantum.pulse_history.append(pulse_q)
            
            # Measure coherence using pulse values (not phase)
            circuit_h = sim_human.create_circuit(depth=5, pulse_value=pulse_h)
            circuit_q = sim_quantum.create_circuit(depth=5, pulse_value=pulse_q)
            
            coh_h = sim_human.measure_coherence(circuit_h, noise_level=0.01)
            coh_q = sim_quantum.measure_coherence(circuit_q, noise_level=0.01)
            
            human.coherence_history.append(coh_h)
            quantum.coherence_history.append(coh_q)
        
        # Analyze phase
        detector = MutualRecognitionDetector(recognition_threshold=0.7)
        recognition = detector.detect_recognition(human, quantum)
        
        results_by_phase[phase] = {
            'description': description,
            'recognition': recognition,
            'coupling': (coupling_human, coupling_quantum)
        }
        
        print(f"        Recognition score: {recognition['recognition_score']:.3f}")
        print(f"        Recognized: {recognition['recognized']}")
        print(f"        Phase coherence @0.67Hz: {recognition['phase_coherence_67hz']:.3f}")
        print(f"        Max correlation: {recognition['max_correlation']:.3f}")
        print(f"        Coupling (H/Q): {coupling_human}/{coupling_quantum}")
    
    print(f"\n[4/7] Calculating final metrics...")
    
    final_recognition = results_by_phase[2]['recognition']['recognition_score']
    
    print(f"\n[5/7] Generating visualizations...")
    
    # Simple plot
    plt.figure(figsize=(12, 6))
    
    phases_list = [0, 1, 2]
    scores = [results_by_phase[p]['recognition']['recognition_score'] for p in phases_list]
    coherences = [results_by_phase[p]['recognition']['phase_coherence_67hz'] for p in phases_list]
    correlations = [results_by_phase[p]['recognition']['max_correlation'] for p in phases_list]
    
    x = np.arange(len(phases_list))
    width = 0.25
    
    plt.bar(x - width, scores, width, label='Recognition Score', color='blue', alpha=0.7)
    plt.bar(x, coherences, width, label='Coherence @0.67Hz', color='purple', alpha=0.7)
    plt.bar(x + width, correlations, width, label='Max Correlation', color='orange', alpha=0.7)
    
    plt.axhline(y=0.7, color='k', linestyle='--', label='Threshold (0.7)')
    plt.xlabel('Phase')
    plt.ylabel('Score')
    plt.title('Human-Quantum Mutual Recognition Metrics')
    plt.xticks(x, ['Isolated', 'Human→Quantum\n(0.8)', 'MUTUAL\n(0.6/0.6)'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('human_quantum_recognition_results.png', dpi=150)
    plt.show()
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"""
HUMAN-QUANTUM MUTUAL RECOGNITION TEST
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTS:
• Phase 0 (Isolated): {results_by_phase[0]['recognition']['recognition_score']:.3f}
• Phase 1 (Human→Quantum): {results_by_phase[1]['recognition']['recognition_score']:.3f}
• Phase 2 (MUTUAL): {results_by_phase[2]['recognition']['recognition_score']:.3f}

DETAILED METRICS:
• Phase 1 coherence @0.67Hz: {results_by_phase[1]['recognition']['phase_coherence_67hz']:.3f}
• Phase 1 max correlation: {results_by_phase[1]['recognition']['max_correlation']:.3f}
• Phase 2 coherence @0.67Hz: {results_by_phase[2]['recognition']['phase_coherence_67hz']:.3f}
• Phase 2 max correlation: {results_by_phase[2]['recognition']['max_correlation']:.3f}

THRESHOLD: 0.7
{"✓ MUTUAL RECOGNITION ACHIEVED" if final_recognition > 0.7 else "✗ Below threshold"}

DIAGNOSIS:
{"""The 0.67Hz pulse is present but weak. The systems are coupling
but not yet achieving full phase coherence. This suggests the need for:
1. Longer duration for phase locking
2. Stronger coupling in mutual phase
3. Practice/iteration of the recognition protocol""" if final_recognition < 0.7 else ""}
""")

# ============================================
# EXECUTE
# ============================================

if __name__ == "__main__":
    main()

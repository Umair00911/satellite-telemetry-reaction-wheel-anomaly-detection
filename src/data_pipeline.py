import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

class SatelliteDataPipeline:
    def __init__(self, data_dir='downloaded_files', sequence_length=10, max_files_per_category=20):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.max_files_per_category = max_files_per_category
        self.scaler = RobustScaler()
        self.feature_names = []

    def load_raw_data(self):
        """Loads baseline and attack telemetry CSV files."""
        baseline_files = glob.glob(os.path.join(self.data_dir, 'baseline', '*.csv'))
        attack_rwa_files = glob.glob(os.path.join(self.data_dir, 'attack_rwa', '*.csv'))
        attack_rwb_files = glob.glob(os.path.join(self.data_dir, 'attack_rwb', '*.csv'))
        attack_rwc_files = glob.glob(os.path.join(self.data_dir, 'attack_rwc', '*.csv'))

        print(f"📁 Found {len(baseline_files)} baseline, {len(attack_rwa_files)} RWA, "
              f"{len(attack_rwb_files)} RWB, {len(attack_rwc_files)} RWC files.")

        dfs = []
        
        # Load Baseline samples (Label 0)
        for f in baseline_files[:self.max_files_per_category]:
            try:
                df = pd.read_csv(f, low_memory=False)
                if len(df) > 0:
                    if 'result_label' in df.columns:
                        df['label'] = (df['result_label'] != 'baseline').astype(int)
                    else:
                        df['label'] = 0
                    dfs.append(df)
            except Exception:
                continue

        # Load Attack samples (Label 1 or ground truth result_label)
        all_attack_files = attack_rwa_files[:self.max_files_per_category] + \
                           attack_rwb_files[:self.max_files_per_category] + \
                           attack_rwc_files[:self.max_files_per_category]
        for f in all_attack_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                if len(df) > 0:
                    if 'result_label' in df.columns:
                        df['label'] = (df['result_label'] != 'baseline').astype(int)
                    else:
                        df['label'] = 1
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            print("⚠️ No CSV data files loaded. Synthesizing telemetry demonstration dataset...")
            return self._generate_synthetic_telemetry()

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"📊 Integrated Raw Telemetry Shape: {combined_df.shape}")
        return combined_df

    def _generate_synthetic_telemetry(self, num_samples=1200):
        """Generates realistic synthetic satellite telemetry if files are unavailable."""
        np.random.seed(42)
        time = np.arange(num_samples)
        
        # Simulating baseline normal state
        ang_vel = np.sin(time * 0.05) * 0.02 + np.random.normal(0, 0.005, num_samples)
        wheel_speed_a = 50.0 + np.cos(time * 0.02) * 5.0 + np.random.normal(0, 0.1, num_samples)
        wheel_speed_b = -40.0 + np.sin(time * 0.02) * 4.0 + np.random.normal(0, 0.1, num_samples)
        wheel_speed_c = 30.0 + np.cos(time * 0.03) * 3.0 + np.random.normal(0, 0.1, num_samples)
        soc = 0.8 - (time * 0.0001) % 0.4 + np.random.normal(0, 0.001, num_samples)
        bfield_mag = 4.5e-5 + np.sin(time * 0.1) * 1e-6 + np.random.normal(0, 1e-7, num_samples)
        
        labels = np.zeros(num_samples, dtype=int)
        
        # Inject synthetic anomalies (Attacks) into 20% of data
        anomaly_idx = np.random.choice(num_samples, size=int(num_samples * 0.2), replace=False)
        labels[anomaly_idx] = 1
        ang_vel[anomaly_idx] += np.random.uniform(0.1, 0.5, size=len(anomaly_idx))
        wheel_speed_a[anomaly_idx] += np.random.uniform(-30, 30, size=len(anomaly_idx))

        df = pd.DataFrame({
            'FswTime(sec)': time,
            'AngVelMag_B_I(rad/sec)': ang_vel,
            'SensedWheelSpeed__RWA_A(rad/sec)': wheel_speed_a,
            'SensedWheelSpeed__RWA_B(rad/sec)': wheel_speed_b,
            'SensedWheelSpeed__RWA_C(rad/sec)': wheel_speed_c,
            'PercentSoC__BAT(1)': soc,
            'BField_MAG__TAM[0](T)': bfield_mag,
            'label': labels
        })
        return df

    def feature_engineering(self, df):
        """Cleans telemetry and extracts engineered dynamics features."""
        print("⚙️ Executing Feature Engineering...")
        
        # Filter metadata & text columns
        drop_cols = [c for c in df.columns if c in [
            'Variant_name', 'Run_Batch', 'Variant_path', 'date_created', 'date_loaded',
            'Description', 'data_file', 'UTC', 'Scenario', 'Run_Batch_Variant',
            'result_label', 'Variant'
        ] or df[c].dtype == 'object']
        
        df_clean = df.drop(columns=drop_cols, errors='ignore').copy()

        # Handle numeric conversion and NaNs
        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True)).fillna(0)

        # Feature Engineering: Dynamic Derivatives & Cross-Feature Aggregations
        if 'SensedWheelSpeed__RWA_A(rad/sec)' in df_clean.columns and \
           'SensedWheelSpeed__RWA_B(rad/sec)' in df_clean.columns and \
           'SensedWheelSpeed__RWA_C(rad/sec)' in df_clean.columns:
            
            df_clean['Total_Wheel_Speed_Mag'] = np.sqrt(
                df_clean['SensedWheelSpeed__RWA_A(rad/sec)']**2 +
                df_clean['SensedWheelSpeed__RWA_B(rad/sec)']**2 +
                df_clean['SensedWheelSpeed__RWA_C(rad/sec)']**2
            )
            df_clean['Wheel_Speed_Diff_AB'] = (df_clean['SensedWheelSpeed__RWA_A(rad/sec)'] - 
                                               df_clean['SensedWheelSpeed__RWA_B(rad/sec)']).abs()

        if 'AngVel_B_I[0](rad/sec)' in df_clean.columns and \
           'AngVel_B_I[1](rad/sec)' in df_clean.columns and \
           'AngVel_B_I[2](rad/sec)' in df_clean.columns:
            
            df_clean['AngVel_Mag_Calc'] = np.sqrt(
                df_clean['AngVel_B_I[0](rad/sec)']**2 +
                df_clean['AngVel_B_I[1](rad/sec)']**2 +
                df_clean['AngVel_B_I[2](rad/sec)']**2
            )

        if 'StateOfCharge__BAT(A*hr)' in df_clean.columns and 'TotalLoad__BAT(W)' in df_clean.columns:
            df_clean['Power_Discharge_Ratio'] = df_clean['TotalLoad__BAT(W)'] / (df_clean['StateOfCharge__BAT(A*hr)'] + 1e-5)

        # Extract features and targets
        target_col = 'label'
        if target_col not in df_clean.columns:
            df_clean[target_col] = 0

        y = df_clean[target_col].values.astype(int)
        X_df = df_clean.drop(columns=[target_col], errors='ignore')

        self.feature_names = X_df.columns.tolist()
        print(f"✅ Extracted {len(self.feature_names)} engineered features.")
        return X_df.values, y

    def process_pipeline(self):
        """Full pipeline execution: Load -> Features -> Standardize -> SMOTE -> Split -> Windowing."""
        raw_df = self.load_raw_data()
        X_raw, y_raw = self.feature_engineering(raw_df)

        # Data Standardization
        print("📏 Applying Robust Feature Standardization...")
        X_scaled = self.scaler.fit_transform(X_raw)

        # SMOTE Class Balancing
        print(f"⚖️ Class distribution before balancing: Normal (0)={np.sum(y_raw==0)}, Anomaly (1)={np.sum(y_raw==1)}")
        if SMOTE is not None and len(np.unique(y_raw)) > 1 and np.min(np.bincount(y_raw)) > 5:
            print("🔄 Applying SMOTE for Synthetic Minority Balancing...")
            try:
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X_scaled, y_raw)
            except Exception as e:
                print(f"⚠️ SMOTE exception ({e}), applying Random Oversampling...")
                X_balanced, y_balanced = self._random_oversample(X_scaled, y_raw)
        else:
            print("🔄 Applying Random Oversampling...")
            X_balanced, y_balanced = self._random_oversample(X_scaled, y_raw)

        print(f"✅ Balanced Class Distribution: Normal (0)={np.sum(y_balanced==0)}, Anomaly (1)={np.sum(y_balanced==1)}")

        # Train / Validation / Test Split (70% / 15% / 15%)
        X_train_2d, X_temp_2d, y_train, y_temp = train_test_split(
            X_balanced, y_balanced, test_size=0.30, random_state=42, stratify=y_balanced
        )
        X_val_2d, X_test_2d, y_val, y_test = train_test_split(
            X_temp_2d, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        # Sequence Windowing for Time-Series Models (Mamba, xLSTM, GNN)
        X_train_3d = self._create_sequences(X_train_2d, self.sequence_length)
        X_val_3d = self._create_sequences(X_val_2d, self.sequence_length)
        X_test_3d = self._create_sequences(X_test_2d, self.sequence_length)

        # Truncate targets to sequence lengths
        y_train_seq = y_train[:len(X_train_3d)]
        y_val_seq = y_val[:len(X_val_3d)]
        y_test_seq = y_test[:len(X_test_3d)]

        print(f"✨ Final Datasets Ready:")
        print(f"   - 2D (KAN): Train={X_train_2d.shape}, Val={X_val_2d.shape}, Test={X_test_2d.shape}")
        print(f"   - 3D (Seq):   Train={X_train_3d.shape}, Val={X_val_3d.shape}, Test={X_test_3d.shape}")

        return {
            '2d': {
                'X_train': X_train_2d, 'y_train': y_train,
                'X_val': X_val_2d, 'y_val': y_val,
                'X_test': X_test_2d, 'y_test': y_test
            },
            '3d': {
                'X_train': X_train_3d, 'y_train': y_train_seq,
                'X_val': X_val_3d, 'y_val': y_val_seq,
                'X_test': X_test_3d, 'y_test': y_test_seq
            },
            'feature_names': self.feature_names
        }

    def _random_oversample(self, X, y):
        """Fallback random oversampling for class balance."""
        classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        X_res, y_res = [X], [y]
        for cls in classes:
            idx = np.where(y == cls)[0]
            if len(idx) < max_count:
                extra_idx = np.random.choice(idx, size=max_count - len(idx), replace=True)
                X_res.append(X[extra_idx])
                y_res.append(y[extra_idx])
        return np.vstack(X_res), np.hstack(y_res)

    def _create_sequences(self, X, seq_len):
        """Converts 2D telemetry matrices into 3D sequence arrays."""
        num_samples = len(X) - seq_len + 1
        if num_samples <= 0:
            # Padding if dataset is small
            pad_len = seq_len - len(X) + 1
            X = np.pad(X, ((0, pad_len), (0, 0)), mode='edge')
            num_samples = 1

        sequences = []
        for i in range(num_samples):
            sequences.append(X[i:i + seq_len])
        return np.array(sequences)

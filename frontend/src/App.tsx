import { useState, useCallback, useRef } from 'react';
import { UploadIcon, CheckCircleIcon, AlertCircleIcon, ImageIcon, TrashIcon, ChartIcon, BrainIcon, DatabaseIcon, ChipIcon } from './Icons';
import './App.css';

interface TopKPrediction {
  class_id: number;
  class_name: string;
  confidence: number;
}

interface PredictionResult {
  class_id: number;
  class_name: string;
  confidence: number;
  is_confident: boolean;
  top_k: TopKPrediction[];
}

const API_URL = 'http://localhost:8000';

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    // Send to API
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFile(file);
    }
  }, [handleFile]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleBrowse = () => {
    fileInputRef.current?.click();
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleClear = () => {
    setImage(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#10b981'; // success
    if (confidence >= 0.5) return '#f59e0b'; // warning
    return '#ef4444'; // error
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.5) return 'medium';
    return 'low';
  };

  return (
    <div className="app">
      <header className="header">
        <h1>
          Traffic Sign Recognition
        </h1>
        <p>Advanced AI-powered identification for Sri Lankan road signs</p>
      </header>

      <main>
        <div className="main-content">
          {/* Left Panel - Image Upload */}
          <section className="panel">
            <div className="panel-header">
              <ImageIcon />
              <h2 className="panel-title">Input Image</h2>
            </div>

            <div
              className={`drop-zone ${isDragging ? 'active' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={handleBrowse}
            >
              {image ? (
                <img src={image} alt="Uploaded" className="preview-image" />
              ) : (
                <>
                  <div className="drop-icon-wrapper">
                    <UploadIcon />
                  </div>
                  <p className="drop-text-primary">Click to upload or drag and drop</p>
                  <p className="drop-text-secondary">SVG, PNG, JPG or GIF (max. 5MB)</p>
                </>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />

            <div className="button-group">
              <button className="btn btn-primary" onClick={handleBrowse}>
                <UploadIcon />
                Select File
              </button>
              {image && (
                <button className="btn btn-secondary" onClick={handleClear}>
                  <TrashIcon />
                  Reset
                </button>
              )}
            </div>
          </section>

          {/* Right Panel - Results */}
          <section className="panel">
            <div className="panel-header">
              <ChartIcon />
              <h2 className="panel-title">Analysis Results</h2>
            </div>

            {loading && (
              <div className="state-container">
                <div className="spinner"></div>
                <p>Analyzing traffic sign features...</p>
              </div>
            )}

            {error && (
              <div className="state-container">
                <div style={{ color: 'var(--error)', marginBottom: '1rem' }}>
                  <AlertCircleIcon />
                </div>
                <h3 style={{ color: 'var(--error)', fontWeight: 600 }}>Analysis Failed</h3>
                <p style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>{error}</p>
              </div>
            )}

            {prediction && !loading && (
              <>
                <div className="prediction-hero">
                  <p className="prediction-label">Detected Sign</p>
                  <div className="prediction-value">{prediction.class_name}</div>
                  <div className={`confidence-badge ${getConfidenceLevel(prediction.confidence)}`}>
                    <CheckCircleIcon />
                    {(prediction.confidence * 100).toFixed(1)}% Confidence
                  </div>
                </div>

                <div className="confidence-list">
                  {prediction.top_k.map((pred, index) => (
                    <div key={index} className="confidence-item">
                      <span className="rank">#{index + 1}</span>
                      <span className="class-name" title={pred.class_name}>
                        {pred.class_name}
                      </span>
                      <div className="bar-wrapper">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${pred.confidence * 100}%`,
                            backgroundColor: getConfidenceColor(pred.confidence),
                          }}
                        />
                      </div>
                      <span className="percentage">
                        {(pred.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}

            {!prediction && !loading && !error && (
              <div className="state-container">
                <div style={{ color: 'var(--text-tertiary)', marginBottom: '1rem' }}>
                  <ImageIcon />
                </div>
                <p>Upload an image to verify traffic sign classification</p>
              </div>
            )}
          </section>
        </div>

        {/* Model Description Section */}
        <section className="model-info-section">
          <div className="info-cards">

            <div className="info-card">
              <div className="info-header">
                <BrainIcon />
                <h3>Model Architecture</h3>
              </div>
              <div className="info-content">
                <p style={{ marginBottom: '1rem' }}>
                  A sophisticated Hybrid Neural Network combining computer vision and attention mechanisms.
                </p>
                <div>
                  <span className="tech-tag">ConvNeXt Tiny Backbone</span>
                  <span className="tech-tag">Transformer Encoder</span>
                  <span className="tech-tag">SE-Block Attention</span>
                  <span className="tech-tag">PyTorch</span>
                </div>
              </div>
            </div>

            <div className="info-card">
              <div className="info-header">
                <DatabaseIcon />
                <h3>Dataset & Training</h3>
              </div>
              <div className="info-content">
                <p style={{ marginBottom: '0.5rem' }}>
                  Trained on a comprehensive synthetic dataset of Sri Lankan road conditions.
                </p>
                <ul style={{ listStyle: 'none', marginTop: '0.5rem' }}>
                  <li>• 125 Unique Classes</li>
                  <li>• ~6,375 Total Images</li>
                  <li>• 100% Scenario Coverage</li>
                  <li>• Exhaustive Permutations</li>
                </ul>
              </div>
            </div>

            <div className="info-card">
              <div className="info-header">
                <ChipIcon />
                <h3>Performance Metrics</h3>
              </div>
              <div className="info-content">
                <p style={{ marginBottom: '0.5rem' }}>
                  State-of-the-art accuracy achieved through rigorous validation.
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
                  <div>
                    <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--success)' }}>99.5%</div>
                    <div style={{ fontSize: '0.8rem' }}>Validation Accuracy</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--accent)' }}>30ms</div>
                    <div style={{ fontSize: '0.8rem' }}>Inference Time</div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </section>
      </main>

      <footer className="footer">
        System Status: Online • Hybrid ConvNeXt-Transformer Model • v2.1
      </footer>
    </div>
  );
}

export default App;

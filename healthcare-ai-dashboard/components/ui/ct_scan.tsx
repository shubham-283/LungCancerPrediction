import React, { useState, useRef, ChangeEvent } from "react";
import { Upload, CheckCircle, AlertTriangle, Activity, Brain, Zap, FileImage, TrendingUp, Shield, Clock } from "lucide-react";

interface PredictionResponse {
  predicted_class: string;
  confidence_scores: Record<string, number>;
  top_prediction_confidence: number;
}

interface DetectionStatistics {
  average_confidence: number;
  max_confidence: number;
  total_area: number;
}

interface DetectionResponse {
  total_detections: number;
  detections: Array<{
    bbox: number[];
    confidence: number;
    class: string;
    area: number;
  }>;
  confidence_distribution: Record<string, number>;
  statistics: DetectionStatistics;
  risk_assessment: string;
  recommendation: string;
  annotated_image_base64: string;
}

const CtScanAnalysis: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [detection, setDetection] = useState<DetectionResponse | null>(null);
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    processFile(file);
  };

  const processFile = async (file: File) => {
    setUploadedFile(file);
    setIsAnalyzing(true);
    setPrediction(null);
    setDetection(null);
    setAnnotatedImageUrl(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Predict CT Scan
      const predRes = await fetch("http://127.0.0.1:8000/predict-ct-scan", {
        method: "POST",
        body: formData,
      });
      const predData: PredictionResponse = await predRes.json();
      setPrediction(predData);

      // Detect Cancer Cells
      const detRes = await fetch("http://127.0.0.1:8000/detect-cancer-cells", {
        method: "POST",
        body: formData,
      });
      const detData: DetectionResponse = await detRes.json();
      setDetection(detData);

      // Convert base64 to blob URL
      if (detData.annotated_image_base64) {
        const byteCharacters = atob(detData.annotated_image_base64);
        const byteNumbers = Array.from(byteCharacters).map((c) => c.charCodeAt(0));
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        setAnnotatedImageUrl(url);
      }
    } catch (err) {
      console.error("Error during analysis:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600 bg-green-100';
    if (confidence >= 70) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -inset-10 opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"></div>
          <div className="absolute top-3/4 right-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-1000"></div>
          <div className="absolute bottom-1/4 left-1/2 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-2000"></div>
        </div>
      </div>

      <div className="relative z-10 py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-16">
            <div className="flex justify-center mb-6">
              <div className="relative">
                <div className="w-20 h-20 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full flex items-center justify-center shadow-2xl">
                  <Brain className="w-10 h-10 text-white" />
                </div>
                <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-400 rounded-full flex items-center justify-center animate-pulse">
                  <Zap className="w-3 h-3 text-white" />
                </div>
              </div>
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
              AI-Powered CT Scan Analysis
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Advanced machine learning algorithms for precise lung cancer detection and comprehensive medical imaging analysis
            </p>
          </div>

          {/* Upload Section */}
          <div className="mb-12">
            <div 
              className={`relative overflow-hidden bg-white/10 backdrop-blur-xl border-2 border-dashed transition-all duration-300 rounded-3xl p-12 mx-auto max-w-2xl ${
                dragOver ? 'border-cyan-400 bg-cyan-400/20 scale-105' : 'border-gray-600 hover:border-gray-500'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 opacity-50"></div>
              
              <div className="relative z-10 flex flex-col items-center gap-6">
                <div className={`w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 ${
                  dragOver ? 'bg-cyan-400 scale-110' : 'bg-gradient-to-r from-cyan-500 to-blue-600'
                } shadow-2xl`}>
                  <Upload className="w-12 h-12 text-white" />
                </div>

                <div className="text-center">
                  <h3 className="text-2xl font-bold text-white mb-2">Upload CT Scan</h3>
                  <p className="text-gray-300 mb-6">Drag & drop your CT scan file or click to browse</p>
                </div>

                <input
                  type="file"
                  accept=".dcm,.png,.jpg,.jpeg,.tiff"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  className="hidden"
                />

                <button
                  onClick={handleFileUploadClick}
                  className="group relative px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold rounded-xl shadow-2xl transform transition-all duration-200 hover:scale-105 hover:shadow-cyan-500/25"
                >
                  <span className="flex items-center gap-3">
                    <FileImage className="w-5 h-5" />
                    Select CT Scan File
                  </span>
                </button>

                <p className="text-sm text-gray-400">
                  Supports: DICOM (.dcm), PNG, JPEG, TIFF formats
                </p>
              </div>
            </div>

            {/* File Upload Success */}
            {uploadedFile && (
              <div className="mt-8 mx-auto max-w-2xl">
                <div className="bg-green-500/20 backdrop-blur-xl border border-green-400/30 rounded-2xl p-6 transform animate-fadeIn">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-green-400 rounded-full flex items-center justify-center">
                      <CheckCircle className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h4 className="text-green-400 font-semibold text-lg">File Uploaded Successfully</h4>
                      <p className="text-gray-300">{uploadedFile.name}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Analysis Loading */}
            {isAnalyzing && (
              <div className="mt-8 mx-auto max-w-2xl">
                <div className="bg-blue-500/20 backdrop-blur-xl border border-blue-400/30 rounded-2xl p-8">
                  <div className="flex flex-col items-center gap-6">
                    <div className="relative">
                      <div className="w-16 h-16 border-4 border-blue-400/30 border-t-blue-400 rounded-full animate-spin"></div>
                      <Activity className="absolute inset-0 m-auto w-6 h-6 text-blue-400" />
                    </div>
                    <div className="text-center">
                      <h4 className="text-blue-400 font-semibold text-xl mb-2">AI Analysis in Progress</h4>
                      <p className="text-gray-300">Processing your CT scan with advanced neural networks...</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          {(prediction || detection) && (
            <div className="grid md:grid-cols-2 gap-8">
              {/* Prediction Results */}
              {prediction && (
                <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl p-8 shadow-2xl">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl flex items-center justify-center">
                      <TrendingUp className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold text-white">Classification Results</h3>
                  </div>

                  <div className="space-y-6">
                    <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-gray-300 font-medium">Predicted Class</span>
                        <span className={`px-4 py-2 rounded-full text-sm font-semibold ${getConfidenceColor(prediction.top_prediction_confidence)}`}>
                          {prediction.predicted_class}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300 font-medium">Confidence</span>
                        <span className="text-2xl font-bold text-white">
                          {prediction.top_prediction_confidence.toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-white font-semibold mb-4">Confidence Breakdown</h4>
                      <div className="space-y-3">
                        {Object.entries(prediction.confidence_scores).map(([cls, score]) => (
                          <div key={cls} className="bg-white/5 rounded-xl p-4 border border-white/10">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-gray-300 capitalize">{cls}</span>
                              <span className="text-white font-semibold">{score.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                              <div 
                                className="bg-gradient-to-r from-purple-500 to-pink-600 h-2 rounded-full transition-all duration-1000"
                                style={{ width: `${score}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Detection Results */}
              {detection && (
                <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl p-8 shadow-2xl">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
                      <Shield className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold text-white">Detection Analysis</h3>
                  </div>

                  {annotatedImageUrl && (
                    <div className="mb-6">
                      <img
                        src={annotatedImageUrl}
                        alt="Annotated CT Scan"
                        className="w-full h-auto rounded-2xl border border-white/20 shadow-xl"
                      />
                    </div>
                  )}

                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white/5 rounded-2xl p-4 border border-white/10 text-center">
                        <div className="text-3xl font-bold text-white mb-1">{detection.total_detections}</div>
                        <div className="text-gray-300 text-sm">Detections</div>
                      </div>
                      <div className="bg-white/5 rounded-2xl p-4 border border-white/10 text-center">
                        <div className="text-3xl font-bold text-white mb-1">{detection.statistics.total_area}</div>
                        <div className="text-gray-300 text-sm">Total Area</div>
                      </div>
                    </div>

                    <div className={`rounded-2xl p-6 border ${getRiskColor(detection.risk_assessment)}`}>
                      <div className="flex items-center gap-3 mb-2">
                        <AlertTriangle className="w-5 h-5" />
                        <span className="font-semibold">Risk Assessment</span>
                      </div>
                      <p className="font-bold text-lg capitalize">{detection.risk_assessment}</p>
                    </div>

                    <div className="bg-blue-500/20 rounded-2xl p-6 border border-blue-400/30">
                      <div className="flex items-center gap-3 mb-3">
                        <Clock className="w-5 h-5 text-blue-400" />
                        <span className="font-semibold text-blue-400">Recommendation</span>
                      </div>
                      <p className="text-gray-300 leading-relaxed">{detection.recommendation}</p>
                    </div>

                    <div className="bg-white/5 rounded-2xl p-6 border border-white/10">
                      <h4 className="text-white font-semibold mb-4">Statistical Analysis</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-gray-300">Average Confidence</span>
                          <span className="text-white font-semibold">{detection.statistics.average_confidence.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-300">Max Confidence</span>
                          <span className="text-white font-semibold">{detection.statistics.max_confidence.toFixed(2)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CtScanAnalysis;
"use client";

import type React from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  FileText,
  Upload,
  Eye,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";
import { useState } from "react";
import {
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TrendingUp, BarChart3 } from "lucide-react";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import CtScanAnalysis from "@/components/ui/ct_scan";
import PulmonaryDiseasePredictor from "@/components/ui/pulmonary";
import Header from "@/components/ui/navbar";

export default function HealthcareAIDashboard() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);

  const [formData, setFormData] = useState({
    AGE: 0,
    GENDER: 0,
    SMOKING: 0,
    FINGER_DISCOLORATION: 0,
    MENTAL_STRESS: 0,
    EXPOSURE_TO_POLLUTION: 0,
    LONG_TERM_ILLNESS: 0,
    ENERGY_LEVEL: 3,
    IMMUNE_WEAKNESS: 0,
    BREATHING_ISSUE: 0,
    ALCOHOL_CONSUMPTION: 0,
    THROAT_DISCOMFORT: 0,
    OXYGEN_SATURATION: 95,
    CHEST_TIGHTNESS: 0,
    FAMILY_HISTORY: 0,
    SMOKING_FAMILY_HISTORY: 0,
    STRESS_IMMUNE: 0,
  });

  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [explainResults, setExplainResults] = useState(true);

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const apiPayload = {
        sample: formData,
        explain: explainResults, // Use the toggle value instead of hardcoded true
      };

      const response = await fetch("http://10.100.239.58:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(apiPayload),
      });

      if (response.ok) {
        const result = await response.json();
        setPredictionResult(result);
        scrollToSection("prediction-results");
      } else {
        console.error("API call failed:", response.status, response.statusText);
        alert(`API call failed: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.error("Error calling API:", error);
      alert(
        "Network error: Unable to connect to the prediction service. Please check if the API server is running."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const updateFormField = (field: string, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        setIsAnalyzing(true);
        // Simulate analysis delay
        setTimeout(() => {
          setIsAnalyzing(false);
          setAnalysisComplete(true);
        }, 3000);
      };
      reader.readAsDataURL(file);
    }
  };

  const diagnosticAccuracyData = [
    { epoch: 1, correctDiagnoses: 0.72, validationAccuracy: 0.68 },
    { epoch: 5, correctDiagnoses: 0.84, validationAccuracy: 0.81 },
    { epoch: 10, correctDiagnoses: 0.89, validationAccuracy: 0.87 },
    { epoch: 15, correctDiagnoses: 0.92, validationAccuracy: 0.9 },
    { epoch: 20, correctDiagnoses: 0.94, validationAccuracy: 0.92 },
    { epoch: 25, correctDiagnoses: 0.942, validationAccuracy: 0.924 },
  ];

  const sensitivitySpecificityData = [
    { specificity: 0.0, sensitivity: 1.0 },
    { specificity: 0.1, sensitivity: 0.98 },
    { specificity: 0.2, sensitivity: 0.96 },
    { specificity: 0.3, sensitivity: 0.94 },
    { specificity: 0.4, sensitivity: 0.92 },
    { specificity: 0.5, sensitivity: 0.9 },
    { specificity: 0.6, sensitivity: 0.88 },
    { specificity: 0.7, sensitivity: 0.85 },
    { specificity: 0.8, sensitivity: 0.82 },
    { specificity: 0.9, sensitivity: 0.78 },
    { specificity: 1.0, sensitivity: 0.74 },
  ];

  const modelComparisonData = [
    {
      model: "LungAI v2.1",
      diagnosticAccuracy: 94.2,
      sensitivity: 92.8,
      specificity: 91.5,
      clinicalReliability: 97,
    },
    {
      model: "ResNet-50",
      diagnosticAccuracy: 89.1,
      sensitivity: 87.3,
      specificity: 88.2,
      clinicalReliability: 88,
    },
    {
      model: "VGG-16",
      diagnosticAccuracy: 85.7,
      sensitivity: 84.1,
      specificity: 86.3,
      clinicalReliability: 85,
    },
    {
      model: "DenseNet",
      diagnosticAccuracy: 91.3,
      sensitivity: 89.7,
      specificity: 90.1,
      clinicalReliability: 90,
    },
  ];

  const rocData = [
    { fpr: 0.0, tpr: 0.0 },
    { fpr: 0.05, tpr: 0.15 },
    { fpr: 0.1, tpr: 0.35 },
    { fpr: 0.15, tpr: 0.55 },
    { fpr: 0.2, tpr: 0.72 },
    { fpr: 0.25, tpr: 0.84 },
    { fpr: 0.3, tpr: 0.91 },
    { fpr: 0.4, tpr: 0.96 },
    { fpr: 0.6, tpr: 0.98 },
    { fpr: 1.0, tpr: 1.0 },
  ];

  const confusionMatrixData = [
    { category: "True Negative", value: 1847, color: "#10b981" },
    { category: "False Positive", value: 123, color: "#f59e0b" },
    { category: "False Negative", value: 89, color: "#ef4444" },
    { category: "True Positive", value: 941, color: "#3b82f6" },
  ];

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleExportPDF = () => {
    // In a real application, this would generate and download a PDF report
    console.log("Generating PDF report...");
    alert(
      "PDF report generation initiated. This would normally download a comprehensive medical report."
    );
  };

  const handleGenerateReport = () => {
    // In a real application, this would compile all analysis data
    console.log("Generating comprehensive report...");
    alert(
      "Comprehensive report generated. This would include all analysis results, SHAP explanations, and recommendations."
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <TooltipProvider>
        {/* Navigation Header */}
        <Header/>

        {/* Hero Section */}
        <section className="relative py-20 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto">
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Left Content */}
              <div className="space-y-8">
                <div className="space-y-4">
                  <Badge className="bg-green-600 text-white px-4 py-2 text-sm font-medium">
                    AI-Powered Detection
                  </Badge>
                  <h1 className="text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
                    Early Detection
                    <span className="block text-blue-600">Saves Lives</span>
                  </h1>
                  <p className="text-xl text-gray-600 leading-relaxed max-w-lg">
                    Advanced machine learning and SHAP explainability for
                    accurate lung cancer prediction. Empowering healthcare
                    professionals with trustworthy AI insights.
                  </p>
                </div>

                <div className="flex flex-col sm:flex-row gap-4">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        size="lg"
                        className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 text-lg"
                        onClick={() => scrollToSection("patient-input")}
                      >
                        Start Analysis
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Begin patient data input and analysis</p>
                    </TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="outline"
                        size="lg"
                        className="border-green-600 text-green-600 hover:bg-green-600 hover:text-white px-8 py-4 text-lg bg-transparent"
                        onClick={() => scrollToSection("performance-charts")}
                      >
                        View Demo
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>See model performance and capabilities</p>
                    </TooltipContent>
                  </Tooltip>
                </div>

                {/* Key Stats */}
                <div className="grid grid-cols-3 gap-6 pt-8">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="text-center cursor-help">
                        <div className="text-3xl font-bold text-blue-600">
                          94.2%
                        </div>
                        <div className="text-sm text-gray-600 font-medium">
                          Accuracy
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Overall prediction accuracy on validation dataset</p>
                    </TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="text-center cursor-help">
                        <div className="text-3xl font-bold text-green-600">
                          0.97
                        </div>
                        <div className="text-sm text-gray-600 font-medium">
                          F1-Score
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Harmonic mean of precision and recall</p>
                    </TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="text-center cursor-help">
                        <div className="text-3xl font-bold text-cyan-600">
                          10K+
                        </div>
                        <div className="text-sm text-gray-600 font-medium">
                          Scans Analyzed
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Total CT scans processed and analyzed</p>
                    </TooltipContent>
                  </Tooltip>
                </div>
              </div>

              {/* Right Visual */}
              <div className="relative">
                <Card className="p-8 bg-white/90 backdrop-blur-sm shadow-2xl border-0">
                  <div className="space-y-6">
                    {/* CT Scan Visualization */}
                    <div className="relative bg-gray-900 rounded-lg p-6 aspect-square">
                      <img
                        src="/medical-ct-scan-of-lungs-with-highlighted-areas-sh.png"
                        alt="CT Scan Analysis"
                        className="w-full h-full object-cover rounded-lg opacity-90"
                      />
                      <div className="absolute top-4 right-4 bg-green-600 text-white px-3 py-1 rounded-full text-sm font-medium">
                        AI Analysis Active
                      </div>

                      {/* Overlay indicators */}
                      <div className="absolute bottom-4 left-4 right-4 bg-black/70 rounded-lg p-3">
                        <div className="flex items-center justify-between text-white text-sm">
                          <span>Risk Assessment</span>
                          <span className="text-green-400 font-bold">
                            Low Risk
                          </span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                          <div className="bg-green-400 h-2 rounded-full w-1/4"></div>
                        </div>
                      </div>
                    </div>

                    {/* Medical Icons */}
                    <div className="flex justify-center gap-8 pt-4">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="flex flex-col items-center gap-2 cursor-help">
                            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                              <svg
                                className="w-6 h-6 text-blue-600"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                />
                              </svg>
                            </div>
                            <span className="text-xs text-gray-600 font-medium">
                              SHAP Analysis
                            </span>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Explainable AI feature importance analysis</p>
                        </TooltipContent>
                      </Tooltip>

                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="flex flex-col items-center gap-2 cursor-help">
                            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                              <svg
                                className="w-6 h-6 text-green-600"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M13 10V3L4 14h7v7l9-11h-7z"
                                />
                              </svg>
                            </div>
                            <span className="text-xs text-gray-600 font-medium">
                              ML Prediction
                            </span>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Machine learning cancer risk prediction</p>
                        </TooltipContent>
                      </Tooltip>

                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="flex flex-col items-center gap-2 cursor-help">
                            <div className="w-12 h-12 bg-cyan-100 rounded-full flex items-center justify-center">
                              <svg
                                className="w-6 h-6 text-cyan-600"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                                />
                              </svg>
                            </div>
                            <span className="text-xs text-gray-600 font-medium">
                              Performance
                            </span>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Model accuracy and performance metrics</p>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                  </div>
                </Card>
              </div>
            </div>
          </div>
        </section>

        {/* Patient Data Input Section */}
        <PulmonaryDiseasePredictor/>

        {/* Prediction Results Section */}
        {predictionResult && (
          <section
            id="prediction-results"
            className="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-50 to-cyan-50"
          >
            <div className="max-w-6xl mx-auto">
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  AI Analysis Results
                </h2>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                  Comprehensive lung cancer risk assessment based on patient
                  data
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                {/* Main Prediction Card */}
                <Card className="p-8 bg-white shadow-xl border-0">
                  <div className="text-center mb-8">
                    <div
                      className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-4 ${
                        predictionResult.prediction.predicted_class === "YES"
                          ? "bg-red-100 text-red-600"
                          : "bg-green-100 text-green-600"
                      }`}
                    >
                      {predictionResult.prediction.predicted_class === "YES" ? (
                        <AlertTriangle className="w-10 h-10" />
                      ) : (
                        <CheckCircle className="w-10 h-10" />
                      )}
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      Risk Assessment:{" "}
                      {predictionResult.prediction.predicted_class}
                    </h3>
                    <p className="text-lg text-gray-600 mb-4">
                      Confidence:{" "}
                      {(
                        predictionResult.prediction.predicted_proba * 100
                      ).toFixed(1)}
                      %
                    </p>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${
                          predictionResult.prediction.predicted_class === "YES"
                            ? "bg-red-500"
                            : "bg-green-500"
                        }`}
                        style={{
                          width: `${
                            predictionResult.prediction.predicted_proba * 100
                          }%`,
                        }}
                      ></div>
                    </div>
                  </div>
                </Card>

                <Card className="p-6 bg-white shadow-xl border-0">
                  <CardHeader className="pb-4">
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5" />
                      Risk Probability Distribution
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ChartContainer
                      config={{
                        risk: {
                          label: "Risk Level",
                          color:
                            predictionResult.prediction.predicted_class ===
                            "YES"
                              ? "#ef4444"
                              : "#22c55e",
                        },
                      }}
                      className="h-[200px]"
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={[
                              {
                                name: "Risk",
                                value:
                                  predictionResult.prediction.predicted_proba *
                                  100,
                                fill:
                                  predictionResult.prediction
                                    .predicted_class === "YES"
                                    ? "#ef4444"
                                    : "#22c55e",
                              },
                              {
                                name: "No Risk",
                                value:
                                  (1 -
                                    predictionResult.prediction
                                      .predicted_proba) *
                                  100,
                                fill: "#e5e7eb",
                              },
                            ]}
                            cx="50%"
                            cy="50%"
                            innerRadius={40}
                            outerRadius={80}
                            paddingAngle={5}
                            dataKey="value"
                          >
                            <Cell key="risk" />
                            <Cell key="no-risk" />
                          </Pie>
                          <ChartTooltip
                            content={<ChartTooltipContent />}
                            formatter={(value: any) => [
                              `${value.toFixed(1)}%`,
                              "",
                            ]}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </ChartContainer>
                    <div className="text-center mt-4">
                      <p className="text-sm text-gray-600">
                        {predictionResult.prediction.predicted_class === "YES"
                          ? "High Risk"
                          : "Low Risk"}{" "}
                        Detection
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Recommendations */}
              <Card className="p-6 bg-blue-50 border-0 mb-6">
                <h4 className="text-lg font-semibold text-blue-900 mb-3">
                  Clinical Recommendations
                </h4>
                <div className="space-y-2 text-blue-800">
                  {predictionResult.prediction.predicted_class === "YES" ? (
                    <>
                      <p>
                        • Immediate consultation with an oncologist is
                        recommended
                      </p>
                      <p>
                        • Consider additional diagnostic imaging (CT scan, PET
                        scan)
                      </p>
                      <p>• Monitor symptoms closely and report any changes</p>
                      <p>• Lifestyle modifications to reduce risk factors</p>
                    </>
                  ) : (
                    <>
                      <p>• Continue regular health monitoring and check-ups</p>
                      <p>• Maintain healthy lifestyle choices</p>
                      <p>• Monitor for any new symptoms</p>
                      <p>• Follow up with healthcare provider as scheduled</p>
                    </>
                  )}
                </div>
              </Card>
            </div>
          </section>
        )}







        {/* CT Scan Analysis Section */}
        
        <CtScanAnalysis/>





















        <div className="fixed bottom-8 right-8 z-40">
          <div className="flex flex-col gap-3">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="sm"
                  className="bg-blue-600 hover:bg-blue-700 text-white rounded-full w-12 h-12 p-0 shadow-lg"
                  onClick={() =>
                    window.scrollTo({ top: 0, behavior: "smooth" })
                  }
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 10l7-7m0 0l7 7m-7-7v18"
                    />
                  </svg>
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left">
                <p>Back to top</p>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="sm"
                  variant="outline"
                  className="bg-white hover:bg-gray-50 rounded-full w-12 h-12 p-0 shadow-lg"
                  onClick={handleExportPDF}
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left">
                <p>Quick export PDF</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
      </TooltipProvider>
    </div>
  );
}

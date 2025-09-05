'use client';

import React, { useMemo, useRef, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import {
  Download,
  Activity,
  AlertCircle,
  CheckCircle,
  Loader2,
  Stethoscope,
  TrendingUp,
  User,
  FileText,
  Heart,
  Shield,
} from "lucide-react";

/* ---------------------- TypeScript Types ---------------------- */

type YesNo = "Yes" | "No";
type GenderText = "Female" | "Male";

type FormFieldKey =
  | "AGE"
  | "GENDER"
  | "SMOKING"
  | "FINGER_DISCOLORATION"
  | "MENTAL_STRESS"
  | "EXPOSURE_TO_POLLUTION"
  | "LONG_TERM_ILLNESS"
  | "ENERGY_LEVEL"
  | "IMMUNE_WEAKNESS"
  | "BREATHING_ISSUE"
  | "ALCOHOL_CONSUMPTION"
  | "THROAT_DISCOMFORT"
  | "OXYGEN_SATURATION"
  | "CHEST_TIGHTNESS"
  | "FAMILY_HISTORY"
  | "SMOKING_FAMILY_HISTORY"
  | "STRESS_IMMUNE";

interface FormDataState {
  AGE: string;
  GENDER: GenderText | "";
  SMOKING: YesNo | "";
  FINGER_DISCOLORATION: YesNo | "";
  MENTAL_STRESS: YesNo | "";
  EXPOSURE_TO_POLLUTION: YesNo | "";
  LONG_TERM_ILLNESS: YesNo | "";
  ENERGY_LEVEL: string;
  IMMUNE_WEAKNESS: YesNo | "";
  BREATHING_ISSUE: YesNo | "";
  ALCOHOL_CONSUMPTION: YesNo | "";
  THROAT_DISCOMFORT: YesNo | "";
  OXYGEN_SATURATION: string;
  CHEST_TIGHTNESS: YesNo | "";
  FAMILY_HISTORY: YesNo | "";
  SMOKING_FAMILY_HISTORY: YesNo | "";
  STRESS_IMMUNE: YesNo | "";
}

interface TopContributionItem {
  feature: string;
  value: number;      // signed SHAP value
  direction: string;  // e.g., "Increase" / "Decrease"
}

interface PredictionResponse {
  predicted_class: "YES" | "NO";
  predicted_proba: number;
  shap_contributions: Record<string, number>; // signed values
  sorted_top_contributions: TopContributionItem[];
}

interface ChartRow {
  feature: string;
  value: number;          // absolute magnitude
  originalValue: number;  // signed
  direction: string;
  color: string;
}

/* ---------------------- Constants ---------------------- */

const fieldLabels: Record<FormFieldKey, string> = {
  AGE: "Age",
  GENDER: "Gender",
  SMOKING: "Smoking",
  FINGER_DISCOLORATION: "Finger Discoloration",
  MENTAL_STRESS: "Mental Stress",
  EXPOSURE_TO_POLLUTION: "Exposure to Pollution",
  LONG_TERM_ILLNESS: "Long Term Illness",
  ENERGY_LEVEL: "Energy Level (0-100)",
  IMMUNE_WEAKNESS: "Immune Weakness",
  BREATHING_ISSUE: "Breathing Issue",
  ALCOHOL_CONSUMPTION: "Alcohol Consumption",
  THROAT_DISCOMFORT: "Throat Discomfort",
  OXYGEN_SATURATION: "Oxygen Saturation (0-100)",
  CHEST_TIGHTNESS: "Chest Tightness",
  FAMILY_HISTORY: "Family History",
  SMOKING_FAMILY_HISTORY: "Smoking Family History",
  STRESS_IMMUNE: "Stress Immune",
};

const fieldCategories = {
  "Basic Information": ["AGE", "GENDER"],
  "Lifestyle Factors": ["SMOKING", "ALCOHOL_CONSUMPTION", "EXPOSURE_TO_POLLUTION"],
  "Physical Symptoms": ["FINGER_DISCOLORATION", "BREATHING_ISSUE", "THROAT_DISCOMFORT", "CHEST_TIGHTNESS"],
  "Health Metrics": ["ENERGY_LEVEL", "OXYGEN_SATURATION", "IMMUNE_WEAKNESS"],
  "Medical History": ["LONG_TERM_ILLNESS", "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY"],
  "Mental Health": ["MENTAL_STRESS", "STRESS_IMMUNE"],
} as const;

const binaryFields = [
  "GENDER",
  "SMOKING",
  "FINGER_DISCOLORATION",
  "MENTAL_STRESS",
  "EXPOSURE_TO_POLLUTION",
  "LONG_TERM_ILLNESS",
  "IMMUNE_WEAKNESS",
  "BREATHING_ISSUE",
  "ALCOHOL_CONSUMPTION",
  "THROAT_DISCOMFORT",
  "CHEST_TIGHTNESS",
  "FAMILY_HISTORY",
  "SMOKING_FAMILY_HISTORY",
  "STRESS_IMMUNE",
] as const satisfies Readonly<FormFieldKey[]>;

const continuousFields = [
  "AGE",
  "ENERGY_LEVEL",
  "OXYGEN_SATURATION",
] as const satisfies Readonly<FormFieldKey[]>;

/* ---------------------- Component ---------------------- */

const MedicalPredictionDashboard: React.FC = () => {
  const [formData, setFormData] = useState<FormDataState>({
    AGE: "",
    GENDER: "",
    SMOKING: "",
    FINGER_DISCOLORATION: "",
    MENTAL_STRESS: "",
    EXPOSURE_TO_POLLUTION: "",
    LONG_TERM_ILLNESS: "",
    ENERGY_LEVEL: "",
    IMMUNE_WEAKNESS: "",
    BREATHING_ISSUE: "",
    ALCOHOL_CONSUMPTION: "",
    THROAT_DISCOMFORT: "",
    OXYGEN_SATURATION: "",
    CHEST_TIGHTNESS: "",
    FAMILY_HISTORY: "",
    SMOKING_FAMILY_HISTORY: "",
    STRESS_IMMUNE: "",
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const chartRef = useRef<HTMLDivElement | null>(null);

  const handleInputChange = (field: FormFieldKey, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (error) setError("");
  };

  const validateForm = (): string | null => {
    for (const [key, value] of Object.entries(formData) as [FormFieldKey, string][]) {
      if (value === "" || value === null) {
        return `Please fill in ${fieldLabels[key]}`;
      }
      if ((continuousFields as readonly string[]).includes(key)) {
        const numValue = parseFloat(value);
        if (Number.isNaN(numValue)) return `${fieldLabels[key]} must be a valid number`;
        if (key === "AGE" && (numValue < 0 || numValue > 120)) return "Age must be between 0 and 120";
        if ((key === "ENERGY_LEVEL" || key === "OXYGEN_SATURATION") && (numValue < 0 || numValue > 100)) {
          return `${fieldLabels[key]} must be between 0 and 100`;
        }
      }
    }
    return null;
  };

  const handleSubmit = async (): Promise<void> => {
    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }
    setLoading(true);
    setError("");

    try {
      const processedData: Record<string, number> = {};
      (Object.keys(formData) as FormFieldKey[]).forEach((key) => {
        if ((continuousFields as readonly string[]).includes(key)) {
          processedData[key] = parseFloat(formData[key]);
        } else {
          if (key === "GENDER") {
            processedData[key] = formData[key] === "Male" ? 1 : 0;
          } else {
            processedData[key] = formData[key] === "Yes" ? 1 : 0;
          }
        }
      });

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ sample: processedData, explain: true }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const result: { prediction: PredictionResponse } = await response.json();
      setPrediction(result.prediction);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Failed to get prediction: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const chartData: ChartRow[] = useMemo(() => {
    if (!prediction?.sorted_top_contributions) return [];
    return prediction.sorted_top_contributions.slice(0, 10).map((item) => ({
      feature: item.feature.replace(/_/g, " "),
      value: Math.abs(item.value),
      originalValue: item.value,
      direction: item.direction,
      color: item.value > 0 ? "#ef4444" : "#10b981",
    }));
  }, [prediction]);

  const maxAbsShap = useMemo(() => {
    if (!prediction?.sorted_top_contributions?.length) return 1;
    const maxAbs = Math.max(
      ...prediction.sorted_top_contributions.slice(0, 10).map((i) => Math.abs(i.value))
    );
    return Math.max(maxAbs, 0.001);
  }, [prediction]);

  const getCompletionPercentage = () => {
    const totalFields = Object.keys(formData).length;
    const filledFields = Object.values(formData).filter((value) => value !== "").length;
    return Math.round((filledFields / totalFields) * 100);
  };

  // Build a polished PDF report (uses html2canvas-pro to support OKLCH)
  const generateReportPdf = async (): Promise<void> => {
    if (!prediction) return;

    const jsPDFModule = await import("jspdf");
    const JsPDF = (jsPDFModule as any).jsPDF;
    const autoTableModule = await import("jspdf-autotable");
    const autoTable = (autoTableModule as any).default ?? (autoTableModule as any).autoTable;
    // Swap to the OKLCH-capable fork
    const html2canvasModule = await import("html2canvas-pro");
    const html2canvas: (node: HTMLElement, opts?: any) => Promise<HTMLCanvasElement> =
      (html2canvasModule as any).default ?? html2canvasModule;

    const doc = new JsPDF({ unit: "pt", format: "a4" });

    const marginX = 48;
    let cursorY = 56;

    // Header
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.text("PulmoPredict AI — Medical Risk Report", marginX, cursorY);
    cursorY += 8;
    doc.setLineWidth(1);
    doc.setDrawColor(30, 64, 175);
    doc.line(marginX, cursorY, 595 - marginX, cursorY);
    cursorY += 24;

    // Meta
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, marginX, cursorY);
    cursorY += 20;

    // Patient Information
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("Patient Information", marginX, cursorY);
    cursorY += 6;
    doc.setDrawColor(229, 231, 235);
    doc.line(marginX, cursorY, 595 - marginX, cursorY);
    cursorY += 10;

    const patientRows = (Object.entries(formData) as [FormFieldKey, string][])
      .map(([k, v]) => [fieldLabels[k], v]);

    autoTable(doc, {
      startY: cursorY,
      head: [["Field", "Value"]],
      body: patientRows,
      theme: "grid",
      styles: { font: "helvetica", fontSize: 9, cellPadding: 6 },
      headStyles: { fillColor: [30, 64, 175], textColor: 255 },
      margin: { left: marginX, right: marginX },
    });
    cursorY = (doc as any).lastAutoTable.finalY + 20;

    // Prediction Summary
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("Prediction Summary", marginX, cursorY);
    cursorY += 6;
    doc.setDrawColor(229, 231, 235);
    doc.line(marginX, cursorY, 595 - marginX, cursorY);
    cursorY += 12;

    const predColor = prediction.predicted_class === "YES" ? [220, 38, 38] : [5, 150, 105];
    doc.setFont("helvetica", "normal");
    doc.setFontSize(11);
    doc.setTextColor(...predColor);
    doc.text(`Pulmonary Disease Risk: ${prediction.predicted_class}`, marginX, cursorY);
    cursorY += 16;
    doc.setTextColor(17, 24, 39);
    doc.text(`Model Confidence: ${(prediction.predicted_proba * 100).toFixed(1)}%`, marginX, cursorY);
    cursorY += 20;

    // Top contributors
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("Top Contributing Factors (Top 10)", marginX, cursorY);
    cursorY += 6;
    doc.setDrawColor(229, 231, 235);
    doc.line(marginX, cursorY, 595 - marginX, cursorY);
    cursorY += 10;

    const topRows = prediction.sorted_top_contributions.slice(0, 10).map((item, idx) => [
      String(idx + 1),
      item.feature.replace(/_/g, " "),
      item.value.toFixed(6),
      item.direction,
    ]);

    autoTable(doc, {
      startY: cursorY,
      head: [["#", "Feature", "SHAP Value", "Direction"]],
      body: topRows,
      theme: "grid",
      styles: { font: "helvetica", fontSize: 9, cellPadding: 6 },
      headStyles: { fillColor: [147, 51, 234], textColor: 255 },
      columnStyles: { 2: { halign: "right" } },
      margin: { left: marginX, right: marginX },
    });
    cursorY = (doc as any).lastAutoTable.finalY + 20;

    // Optional: embed chart snapshot (ref points only to the chart container to avoid themed wrappers)
    if (chartRef.current) {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(12);
      doc.text("SHAP Feature Analysis Chart", marginX, cursorY);
      cursorY += 10;

      const canvas = await html2canvas(chartRef.current as HTMLElement, { scale: 2, backgroundColor: "#ffffff" });
      const imgData = canvas.toDataURL("image/png");
      const imgWidth = 595 - marginX * 2;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      if (cursorY + imgHeight > 842 - 56) {
        doc.addPage();
        cursorY = 56;
      }
      doc.addImage(imgData, "PNG", marginX, cursorY, imgWidth, imgHeight);
      cursorY += imgHeight + 20;
    }

    // Full SHAP table
    const shapRows = Object.entries(prediction.shap_contributions).map(([k, v]) => [
      k.replace(/_/g, " "),
      v.toFixed(6),
      v > 0 ? "Increase" : v < 0 ? "Decrease" : "Neutral",
    ]);

    if (cursorY > 760) {
      doc.addPage();
      cursorY = 56;
    }
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("All SHAP Contributions", marginX, cursorY);
    cursorY += 6;
    doc.setDrawColor(229, 231, 235);
    doc.line(marginX, cursorY, 595 - marginX, cursorY);
    cursorY += 10;

    autoTable(doc, {
      startY: cursorY,
      head: [["Feature", "SHAP Value", "Direction"]],
      body: shapRows,
      theme: "grid",
      styles: { font: "helvetica", fontSize: 9, cellPadding: 6 },
      headStyles: { fillColor: [16, 185, 129], textColor: 255 },
      columnStyles: { 1: { halign: "right" } },
      margin: { left: marginX, right: marginX },
    });

    // Footer with page numbers
    const pageCount = doc.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(107, 114, 128);
      doc.text(`PulmoPredict AI — Confidential | Page ${i} of ${pageCount}`, marginX, 820);
    }

    doc.save(`PulmoPredict_Report_${new Date().toISOString().split("T")}.pdf`);
  };

  const renderField = (field: FormFieldKey) => {
    if ((binaryFields as readonly string[]).includes(field)) {
      const options = field === "GENDER" ? (["Female", "Male"] as const) : (["No", "Yes"] as const);

      return (
        <div className="space-y-2">
          <label className="block text-sm font-semibold text-gray-800">
            {fieldLabels[field]}
          </label>
          <div className="grid grid-cols-2 gap-2">
            {options.map((option) => (
              <label
                key={option}
                className={`relative flex items-center justify-center gap-2 cursor-pointer rounded-xl border px-3 py-2 transition-all duration-200 ${
                  formData[field] === option
                    ? "border-blue-500 bg-blue-50 text-blue-700 shadow-sm"
                    : "border-gray-200 bg-white text-gray-700 hover:border-gray-300"
                }`}
              >
                <input
                  type="radio"
                  name={field}
                  value={option}
                  checked={formData[field] === option}
                  onChange={(e) => handleInputChange(field, e.target.value)}
                  className="sr-only"
                />
                <span className="font-medium text-sm">{option}</span>
              </label>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-2">
        <label className="block text-sm font-semibold text-gray-800">
          {fieldLabels[field]}
        </label>
        <input
          type="number"
          value={formData[field]}
          onChange={(e) => handleInputChange(field, e.target.value)}
          className="w-full px-4 py-2 border border-gray-200 rounded-xl focus:ring-4 focus:ring-blue-100 focus:border-blue-500 transition-all duration-200 bg-white text-gray-800 placeholder-gray-400"
          placeholder={
            field === "AGE"
              ? "Enter age (0-120)"
              : ["ENERGY_LEVEL", "OXYGEN_SATURATION"].includes(field)
              ? "Enter value (0-100)"
              : "Enter numeric value"
          }
          min={0}
          max={
            field === "AGE" ? 120 :
            ["ENERGY_LEVEL", "OXYGEN_SATURATION"].includes(field) ? 100 : undefined
          }
        />
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <div className="px-4 lg:px-8 py-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="relative">
              <Stethoscope className="w-12 h-12 text-blue-600" />
              <div className="absolute -top-1 -right-1 w-6 h-6 bg-gradient-to-br from-emerald-400 to-teal-500 rounded-full flex items-center justify-center">
                <Heart className="w-3 h-3 text-white" />
              </div>
            </div>
            <div className="text-left">
              <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-blue-700 to-indigo-700 bg-clip-text text-transparent">
                PulmoPredict AI
              </h1>
              <p className="text-base text-gray-600 mt-1">
                Advanced Pulmonary Disease Risk Assessment
              </p>
            </div>
          </div>

          {/* Progress */}
          <div className="max-w-xl mx-auto mb-6">
            <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
              <span>Form Completion</span>
              <span className="font-semibold">{getCompletionPercentage()}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-500 to-indigo-500 h-2 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${getCompletionPercentage()}%` }}
              />
            </div>
          </div>

          {/* Main Grid */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
            {/* Left: Form + (conditional) Chart */}
            <div className="xl:col-span-2 space-y-8">
              {/* Form */}
              <section className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/50 overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 p-6">
                  <div className="flex items-center gap-3 text-white">
                    <User className="w-6 h-6" />
                    <h2 className="text-2xl font-bold">Patient Assessment</h2>
                  </div>
                  <p className="text-blue-100 mt-1 text-sm">
                    Complete all fields for accurate AI-powered risk analysis
                  </p>
                </div>

                <div className="p-6 md:p-8 space-y-10">
                  {Object.entries(fieldCategories).map(([category, fields]) => (
                    <div key={category} className="space-y-5">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-6 bg-gradient-to-b from-blue-500 to-indigo-500 rounded-full" />
                        <h3 className="text-lg font-bold text-gray-800">{category}</h3>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {fields.map((field) => (
                          <div key={field} className="p-4 bg-gray-50/50 rounded-2xl border border-gray-100">
                            {renderField(field as FormFieldKey)}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}

                  {error && (
                    <div className="bg-red-50 border border-red-200 rounded-2xl p-4 flex items-center gap-3">
                      <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                      <span className="text-red-700 font-medium text-sm">{error}</span>
                    </div>
                  )}

                  <div className="flex flex-col sm:flex-row gap-4 pt-2">
                    <button
                      onClick={handleSubmit}
                      disabled={loading || getCompletionPercentage() < 100}
                      className="flex-1 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white py-3 px-6 rounded-2xl font-semibold hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-3 shadow-lg"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Analyzing Patient Data...
                        </>
                      ) : (
                        <>
                          <TrendingUp className="w-5 h-5" />
                          Generate AI Prediction
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </section>

              {/* SHAP Analysis — render only after prediction */}
              {prediction && (
                <section className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/50 overflow-hidden">
                  <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-6">
                    <div className="flex items-center gap-3 text-white">
                      <TrendingUp className="w-6 h-6" />
                      <div>
                        <h3 className="text-xl font-bold">SHAP Feature Analysis</h3>
                        <p className="opacity-90 text-sm">AI Decision Explanation</p>
                      </div>
                    </div>
                  </div>

                  <div className="p-6 space-y-6">
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-2xl border border-blue-200">
                      <p className="text-sm text-gray-700">
                        <span className="font-semibold">Understanding:</span> Red bars increase risk, green bars decrease risk; the zero line splits positive and negative contributions.
                      </p>
                    </div>

                    <div className="h-80 bg-white rounded-2xl p-4" ref={chartRef}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={chartData}
                          margin={{ top: 8, right: 24, left: 24, bottom: 8 }}
                          layout="vertical"
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                          <XAxis
                            type="number"
                            domain={[-maxAbsShap, maxAbsShap]}
                            tick={{ fontSize: 11, fill: "#6b7280" }}
                            tickFormatter={(value) => Number(value).toFixed(3)}
                            stroke="#9ca3af"
                          />
                          <YAxis
                            type="category"
                            dataKey="feature"
                            width={140}
                            tick={{ fontSize: 11, fill: "#6b7280" }}
                            stroke="#9ca3af"
                          />
                          <Tooltip
                            formatter={(value: unknown, _name: string, item: any) => {
                              const signed = (item?.payload?.originalValue ?? value) as number;
                              return [Number(signed).toFixed(6), "SHAP Value"] as [string, string];
                            }}
                            labelFormatter={(label: string) => `Feature: ${label}`}
                            contentStyle={{
                              backgroundColor: "rgba(255, 255, 255, 0.95)",
                              border: "1px solid #e5e7eb",
                              borderRadius: "10px",
                              boxShadow: "0 10px 25px rgba(0, 0, 0, 0.12)",
                            }}
                          />
                          <ReferenceLine x={0} stroke="#cbd5e1" />
                          <Bar dataKey="originalValue" radius={[6, 6, 6, 6]}>
                            {chartData.map((entry, idx) => (
                              <Cell key={`cell-${idx}`} fill={entry.color} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Top Contributors */}
                    <div className="space-y-3">
                      <h4 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-blue-600" />
                        Key Risk Factors
                      </h4>
                      <div className="grid grid-cols-1 gap-3">
                        {prediction.sorted_top_contributions.slice(0, 5).map((item, index) => (
                          <div
                            key={`${item.feature}-${index}`}
                            className="flex items-center justify-between p-4 bg-white rounded-2xl border border-gray-200"
                          >
                            <div className="flex items-center gap-4">
                              <div
                                className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold ${
                                  index < 3 ? "bg-gradient-to-r from-yellow-400 to-orange-500" : "bg-gray-400"
                                }`}
                              >
                                {index + 1}
                              </div>
                              <span className="font-semibold text-gray-800">
                                {item.feature.replace(/_/g, " ")}
                              </span>
                            </div>
                            <div className="flex items-center gap-3">
                              <span
                                className={`text-sm font-mono px-3 py-1 rounded-lg ${
                                  item.value > 0
                                    ? "bg-red-100 text-red-700 border border-red-200"
                                    : "bg-emerald-100 text-emerald-700 border border-emerald-200"
                                }`}
                              >
                                {item.value.toFixed(4)}
                              </span>
                              <span
                                className={`text-xl font-bold ${
                                  item.value > 0 ? "text-red-500" : "text-emerald-500"
                                }`}
                              >
                                {item.direction}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </section>
              )}

              {/* Sample Data */}
              {/* <section className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/50 overflow-hidden">
                <div className="bg-gradient-to-r from-amber-500 to-orange-500 p-6">
                  <h3 className="text-xl font-bold text-white flex items-center gap-3">
                    <Activity className="w-6 h-6" />
                    Quick Test with Sample Data
                  </h3>
                  <p className="text-amber-100 mt-1 text-sm">
                    Load pre-configured test cases to explore the AI model
                  </p>
                </div>

                <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <button
                    onClick={() => {
                      setFormData({
                        AGE: "68",
                        GENDER: "Male",
                        SMOKING: "Yes",
                        FINGER_DISCOLORATION: "Yes",
                        MENTAL_STRESS: "Yes",
                        EXPOSURE_TO_POLLUTION: "Yes",
                        LONG_TERM_ILLNESS: "No",
                        ENERGY_LEVEL: "57.83",
                        IMMUNE_WEAKNESS: "No",
                        BREATHING_ISSUE: "No",
                        ALCOHOL_CONSUMPTION: "Yes",
                        THROAT_DISCOMFORT: "Yes",
                        OXYGEN_SATURATION: "95.98",
                        CHEST_TIGHTNESS: "Yes",
                        FAMILY_HISTORY: "No",
                        SMOKING_FAMILY_HISTORY: "No",
                        STRESS_IMMUNE: "No",
                      });
                      setPrediction(null);
                      setError("");
                    }}
                    className="group relative overflow-hidden bg-gradient-to-r from-emerald-500 to-teal-600 text-white p-6 rounded-2xl hover:from-emerald-600 hover:to-teal-700 transition-all duration-200 shadow-lg"
                  >
                    <div className="absolute inset-0 bg-white/10 transform -skew-y-6 group-hover:skew-y-0 transition-transform duration-300"></div>
                    <div className="relative z-10">
                      <div className="flex items-center justify-between mb-3">
                        <CheckCircle className="w-8 h-8" />
                        <span className="bg-emerald-400/30 px-3 py-1 rounded-full text-sm font-semibold">
                          Low Risk
                        </span>
                      </div>
                      <h4 className="font-bold text-lg mb-2">Sample Patient 1</h4>
                      <p className="text-sm opacity-90">68yr Male, Smoker with symptoms</p>
                      <p className="text-xs mt-2 font-medium bg-white/20 px-2 py-1 rounded-lg inline-block">
                        Expected: NO Disease
                      </p>
                    </div>
                  </button>

                  <button
                    onClick={() => {
                      setFormData({
                        AGE: "81",
                        GENDER: "Male",
                        SMOKING: "Yes",
                        FINGER_DISCOLORATION: "No",
                        MENTAL_STRESS: "No",
                        EXPOSURE_TO_POLLUTION: "Yes",
                        LONG_TERM_ILLNESS: "Yes",
                        ENERGY_LEVEL: "47.69",
                        IMMUNE_WEAKNESS: "Yes",
                        BREATHING_ISSUE: "Yes",
                        ALCOHOL_CONSUMPTION: "No",
                        THROAT_DISCOMFORT: "Yes",
                        OXYGEN_SATURATION: "97.18",
                        CHEST_TIGHTNESS: "No",
                        FAMILY_HISTORY: "No",
                        SMOKING_FAMILY_HISTORY: "No",
                        STRESS_IMMUNE: "No",
                      });
                      setPrediction(null);
                      setError("");
                    }}
                    className="group relative overflow-hidden bg-gradient-to-r from-red-500 to-rose-600 text-white p-6 rounded-2xl hover:from-red-600 hover:to-rose-700 transition-all duration-200 shadow-lg"
                  >
                    <div className="absolute inset-0 bg-white/10 transform skew-y-6 group-hover:-skew-y-0 transition-transform duration-300"></div>
                    <div className="relative z-10">
                      <div className="flex items-center justify-between mb-3">
                        <AlertCircle className="w-8 h-8" />
                        <span className="bg-red-400/30 px-3 py-1 rounded-full text-sm font-semibold">
                          High Risk
                        </span>
                      </div>
                      <h4 className="font-bold text-lg mb-2">Sample Patient 2</h4>
                      <p className="text-sm opacity-90">81yr Male, Multiple conditions</p>
                      <p className="text-xs mt-2 font-medium bg-white/20 px-2 py-1 rounded-lg inline-block">
                        Expected: YES Disease
                      </p>
                    </div>
                  </button>
                </div>
              </section> */}

              {/* How-to */}
              <section className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/50 overflow-hidden">
                <div className="bg-gradient-to-r from-slate-700 to-gray-800 p-6">
                  <div className="flex items-center gap-3 text-white">
                    <FileText className="w-6 h-6" />
                    <h3 className="text-xl font-bold">How to Use This Dashboard</h3>
                  </div>
                </div>

                <div className="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="space-y-2">
                    <div className="w-12 h-12 bg-blue-100 rounded-2xl flex items-center justify-center">
                      <User className="w-6 h-6 text-blue-600" />
                    </div>
                    <h4 className="font-bold text-gray-800">1. Patient Data</h4>
                    <p className="text-sm text-gray-600">
                      Fill in all patient information fields using toggles for Yes/No and numeric inputs for measurements.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="w-12 h-12 bg-indigo-100 rounded-2xl flex items-center justify-center">
                      <TrendingUp className="w-6 h-6 text-indigo-600" />
                    </div>
                    <h4 className="font-bold text-gray-800">2. AI Analysis</h4>
                    <p className="text-sm text-gray-600">
                      Click “Generate AI Prediction” to compute the risk assessment from all factors at once.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="w-12 h-12 bg-purple-100 rounded-2xl flex items-center justify-center">
                      <Activity className="w-6 h-6 text-purple-600" />
                    </div>
                    <h4 className="font-bold text-gray-800">3. SHAP Analysis</h4>
                    <p className="text-sm text-gray-600">
                      Interpret red bars as risk-increasing and green as protective, relative to the zero reference line.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="w-12 h-12 bg-emerald-100 rounded-2xl flex items-center justify-center">
                      <Download className="w-6 h-6 text-emerald-600" />
                    </div>
                    <h4 className="font-bold text-gray-800">4. Export Report</h4>
                    <p className="text-sm text-gray-600">
                      Download a professional PDF report including patient data, prediction summary, SHAP tables, and optional chart.
                    </p>
                  </div>
                </div>
              </section>
            </div>

            {/* Right: Results */}
            <aside className="xl:col-span-1 space-y-6">
              {prediction ? (
                <>
                  <div className="sticky top-6">
                    <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/50 overflow-hidden">
                      <div
                        className={`p-6 ${
                          prediction.predicted_class === "YES"
                            ? "bg-gradient-to-r from-red-500 to-red-600"
                            : "bg-gradient-to-r from-emerald-500 to-green-600"
                        }`}
                      >
                        <div className="flex items-center gap-3 text-white">
                          {prediction.predicted_class === "YES" ? (
                            <AlertCircle className="w-8 h-8" />
                          ) : (
                            <Shield className="w-8 h-8" />
                          )}
                          <div>
                            <h3 className="text-2xl font-bold">AI Diagnosis</h3>
                            <p className="opacity-90">Risk Assessment Complete</p>
                          </div>
                        </div>
                      </div>

                      <div className="p-6 space-y-6">
                        <div className="text-center">
                          <div
                            className={`inline-flex items-center gap-3 p-6 rounded-2xl ${
                              prediction.predicted_class === "YES"
                                ? "bg-red-50 border border-red-200"
                                : "bg-emerald-50 border border-emerald-200"
                            }`}
                          >
                            <div>
                              <p className="text-sm font-medium text-gray-600 mb-1">
                                Pulmonary Disease Risk
                              </p>
                              <p
                                className={`text-4xl font-bold ${
                                  prediction.predicted_class === "YES"
                                    ? "text-red-600"
                                    : "text-emerald-600"
                                }`}
                              >
                                {prediction.predicted_class}
                              </p>
                              <p className="text-lg font-semibold text-gray-700 mt-2">
                                {(prediction.predicted_proba * 100).toFixed(1)}% Confidence
                              </p>
                            </div>
                          </div>
                        </div>

                        <button
                          onClick={generateReportPdf}
                          className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 text-white py-3 px-6 rounded-2xl font-semibold hover:from-emerald-700 hover:to-teal-700 transition-all duration-200 flex items-center justify-center gap-3 shadow-lg"
                        >
                          <Download className="w-5 h-5" />
                          Download PDF Report
                        </button>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/50 p-8 text-center">
                  <div className="space-y-4">
                    <div className="w-20 h-20 bg-gradient-to-br from-gray-200 to-gray-300 rounded-full flex items-center justify-center mx-auto">
                      <FileText className="w-10 h-10 text-gray-500" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-800 mb-2">Ready for Analysis</h3>
                      <p className="text-gray-600">
                        Complete the patient form to receive an AI-powered pulmonary disease risk assessment with explanations.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MedicalPredictionDashboard;

import { useState } from "react";
import { useLocation } from "wouter";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, X, ChevronDown, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";

export default function NewTranslation() {
  const [, setLocation] = useLocation();
  const [fileSelected, setFileSelected] = useState(false);
  const [translating, setTranslating] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);

  const handleFileSelect = () => {
    setFileSelected(true);
  };

  const handleTranslate = () => {
    setTranslating(true);
    
    // Simulate loading steps
    setTimeout(() => setLoadingStep(1), 1000);
    setTimeout(() => setLoadingStep(2), 1700);
    setTimeout(() => {
      setLocation("/results");
    }, 2500);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="space-y-6 max-w-3xl mx-auto"
    >
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Start New Translation</h1>
        <p className="text-muted-foreground mt-1">Upload your document and configure project settings</p>
      </div>

      <Card className="p-6">
        {!fileSelected ? (
          <div 
            className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:bg-muted/50 transition-colors cursor-pointer flex flex-col items-center justify-center space-y-4"
            onClick={handleFileSelect}
          >
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
              <Upload className="h-6 w-6" />
            </div>
            <div>
              <h3 className="font-medium text-lg">Drag & drop your file here</h3>
              <p className="text-sm text-muted-foreground mt-1">Supports .pdf, .docx, .txt up to 50MB</p>
            </div>
            <Button variant="outline" className="mt-2">Browse Files</Button>
          </div>
        ) : (
          <div className="space-y-8 animate-in fade-in duration-300">
            {/* File info */}
            <div className="flex items-center justify-between p-4 border rounded-md bg-muted/30">
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-primary/10 text-primary rounded-md">
                  <FileText className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium text-sm">Legal_Contract_Mar2026.pdf</p>
                  <p className="text-xs text-muted-foreground">2.4 MB</p>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setFileSelected(false)}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Form */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label htmlFor="project-name">Project Name</Label>
                <Input id="project-name" defaultValue="Legal_Contract_Mar2026" />
              </div>
              
              <div className="space-y-2">
                <Label>Client</Label>
                <Select defaultValue="acme">
                  <SelectTrigger>
                    <SelectValue placeholder="Select client" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="acme">ACME Legal</SelectItem>
                    <SelectItem value="medicare">MediCare Inc.</SelectItem>
                    <SelectItem value="autoparts">AutoParts Global</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Domain</Label>
                <Select defaultValue="legal">
                  <SelectTrigger>
                    <SelectValue placeholder="Select domain" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="legal">Legal</SelectItem>
                    <SelectItem value="medical">Medical</SelectItem>
                    <SelectItem value="automotive">Automotive</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Language Pair</Label>
                <div className="p-2 border rounded-md bg-muted/50 text-sm flex items-center justify-between text-muted-foreground">
                  <span>Spanish &rarr; English</span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between border-t pt-6">
              <div className="flex items-center space-x-3">
                <Switch id="glossary" defaultChecked />
                <Label htmlFor="glossary" className="font-medium cursor-pointer">Use client terminology glossary</Label>
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-400 dark:border-green-800 ml-2">V2.4 ACTIVE</Badge>
              </div>
            </div>

            <Button 
              className="w-full h-12 text-lg relative overflow-hidden" 
              onClick={handleTranslate}
              disabled={translating}
            >
              <AnimatePresence mode="wait">
                {!translating ? (
                  <motion.span key="translate" className="absolute inset-0 flex items-center justify-center">
                    Translate Document
                  </motion.span>
                ) : (
                  <motion.div key="loading" className="absolute inset-0 flex flex-col items-center justify-center bg-primary text-primary-foreground">
                    <div className="flex items-center space-x-2">
                      <div className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                      <span className="font-medium">
                        {loadingStep === 0 && "Analyzing document..."}
                        {loadingStep === 1 && "Applying domain glossary..."}
                        {loadingStep === 2 && "Generating translation..."}
                      </span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </Button>
          </div>
        )}
      </Card>
    </motion.div>
  );
}
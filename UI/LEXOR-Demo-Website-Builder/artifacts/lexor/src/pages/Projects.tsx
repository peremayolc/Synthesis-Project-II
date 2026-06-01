import { useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { projects } from "@/data/fakeData";
import { Search, Filter, Plus, FileText, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function Projects() {
  const [, setLocation] = useLocation();
  const [searchTerm, setSearchTerm] = useState("");
  const [clientFilter, setClientFilter] = useState("all");

  const filteredProjects = projects.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          p.client.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesClient = clientFilter === "all" || p.client === clientFilter;
    return matchesSearch && matchesClient;
  });

  const getStatusBadge = (status: string, color: string) => {
    let className = "";
    if (color === "amber") className = "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-800";
    if (color === "green") className = "bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800";
    if (color === "blue") className = "bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800";

    return <Badge className={`font-medium ${className}`}>{status}</Badge>;
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Projects</h1>
          <p className="text-muted-foreground mt-1">Manage and track your translation history</p>
        </div>
        <Button onClick={() => setLocation("/")} className="gap-2">
          <Plus className="w-4 h-4" /> New Translation
        </Button>
      </div>

      <Card className="p-4">
        <div className="flex flex-col sm:flex-row gap-4 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input 
              placeholder="Search projects..." 
              className="pl-9"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <Select value={clientFilter} onValueChange={setClientFilter}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Client" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Clients</SelectItem>
              <SelectItem value="ACME Legal">ACME Legal</SelectItem>
              <SelectItem value="MediCare Inc.">MediCare Inc.</SelectItem>
              <SelectItem value="AutoParts Global">AutoParts Global</SelectItem>
            </SelectContent>
          </Select>
          <Select defaultValue="all">
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Domain" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Domains</SelectItem>
              <SelectItem value="legal">Legal</SelectItem>
              <SelectItem value="medical">Medical</SelectItem>
              <SelectItem value="automotive">Automotive</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="border rounded-md">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Project Name</TableHead>
                <TableHead>Client</TableHead>
                <TableHead>Domain</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredProjects.map((project) => (
                <TableRow 
                  key={project.id} 
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => setLocation("/results")}
                >
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      {project.name}
                    </div>
                  </TableCell>
                  <TableCell>{project.client}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className="font-normal">{project.domain}</Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{project.date}</TableCell>
                  <TableCell>{getStatusBadge(project.status, project.statusColor)}</TableCell>
                  <TableCell className="text-right">
                    <ChevronRight className="w-4 h-4 text-muted-foreground inline-block" />
                  </TableCell>
                </TableRow>
              ))}
              {filteredProjects.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                    No projects found
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </Card>
    </motion.div>
  );
}
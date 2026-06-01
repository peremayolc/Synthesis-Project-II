import { useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { clients as initialClients } from "@/data/fakeData";
import { Plus, Building2, FolderKanban, BookOpen, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog";

export default function Clients() {
  const [, setLocation] = useLocation();
  const [clients, setClients] = useState(initialClients);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const [newClientName, setNewClientName] = useState("");
  const [newClientDomain, setNewClientDomain] = useState("Legal");
  const [newClientGlossary, setNewClientGlossary] = useState(true);

  const handleAddClient = () => {
    if (!newClientName) return;
    const newClient = {
      id: Math.random().toString(),
      name: newClientName,
      domain: newClientDomain,
      glossaryActive: newClientGlossary,
      projectsCount: 0
    };
    setClients([newClient, ...clients]);
    setIsModalOpen(false);
    setNewClientName("");
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Clients</h1>
          <p className="text-muted-foreground mt-1">Manage clients and terminology glossaries</p>
        </div>
        
        <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
          <DialogTrigger asChild>
            <Button className="gap-2">
              <Plus className="w-4 h-4" /> Add Client
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Add New Client</DialogTitle>
            </DialogHeader>
            <div className="grid gap-6 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Client Name</Label>
                <Input 
                  id="name" 
                  value={newClientName}
                  onChange={(e) => setNewClientName(e.target.value)}
                  placeholder="e.g. Global Tech Corp" 
                />
              </div>
              <div className="space-y-2">
                <Label>Primary Domain</Label>
                <Select value={newClientDomain} onValueChange={setNewClientDomain}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select domain" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Legal">Legal</SelectItem>
                    <SelectItem value="Medical">Medical</SelectItem>
                    <SelectItem value="Automotive">Automotive</SelectItem>
                    <SelectItem value="Technology">Technology</SelectItem>
                    <SelectItem value="Financial">Financial</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center justify-between border-t pt-4">
                <div className="space-y-0.5">
                  <Label className="text-base">Terminology Glossary</Label>
                  <p className="text-sm text-muted-foreground">Does this client have an active glossary?</p>
                </div>
                <Switch 
                  checked={newClientGlossary}
                  onCheckedChange={setNewClientGlossary}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsModalOpen(false)}>Cancel</Button>
              <Button onClick={handleAddClient}>Add Client</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {clients.map((client) => (
          <Card key={client.id} className="p-6 flex flex-col h-full hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-4">
              <div className="p-3 bg-primary/10 rounded-lg text-primary">
                <Building2 className="w-6 h-6" />
              </div>
              <Badge variant="outline" className="font-medium bg-muted/50">{client.domain}</Badge>
            </div>
            
            <div className="mb-6">
              <h3 className="text-xl font-bold mb-1">{client.name}</h3>
              <div className="flex items-center text-sm text-muted-foreground mt-2">
                <FolderKanban className="w-4 h-4 mr-2" />
                {client.projectsCount} total projects
              </div>
            </div>

            <div className="mt-auto space-y-4">
              <div className="p-3 border rounded-md bg-muted/30 flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <BookOpen className="w-4 h-4 text-muted-foreground" /> Glossary
                </div>
                {client.glossaryActive ? (
                  <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800">
                    <CheckCircle2 className="w-3 h-3 mr-1" /> Active
                  </Badge>
                ) : (
                  <Badge className="bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-800">
                    <AlertCircle className="w-3 h-3 mr-1" /> Missing
                  </Badge>
                )}
              </div>
              
              <Button 
                variant="outline" 
                className="w-full"
                onClick={() => setLocation("/projects")}
              >
                View Projects
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </motion.div>
  );
}
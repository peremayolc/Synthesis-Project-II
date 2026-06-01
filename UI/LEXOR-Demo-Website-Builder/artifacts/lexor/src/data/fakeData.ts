export const clients = [
  { id: '1', name: 'ACME Legal', domain: 'Legal', glossaryActive: true, projectsCount: 12 },
  { id: '2', name: 'MediCare Inc.', domain: 'Medical', glossaryActive: true, projectsCount: 8 },
  { id: '3', name: 'AutoParts Global', domain: 'Automotive', glossaryActive: false, projectsCount: 5 },
];

export const projects = [
  { id: '1', name: 'Legal_Contract_Mar2026', client: 'ACME Legal', domain: 'Legal', date: 'Today', status: 'Needs Review', statusColor: 'amber' },
  { id: '2', name: 'Medical_Device_Manual', client: 'MediCare Inc.', domain: 'Medical', date: 'Yesterday', status: 'Auto-accepted', statusColor: 'green' },
  { id: '3', name: 'Engine_Specs_v3', client: 'AutoParts Global', domain: 'Automotive', date: 'Mar 1', status: 'Sent to Human', statusColor: 'blue' },
  { id: '4', name: 'Terms_of_Service_Update', client: 'ACME Legal', domain: 'Legal', date: 'Feb 28', status: 'Auto-accepted', statusColor: 'green' },
  { id: '5', name: 'Clinical_Trial_Results', client: 'MediCare Inc.', domain: 'Medical', date: 'Feb 25', status: 'Needs Review', statusColor: 'amber' },
];

export const initialSegments = [
  { id: 1, source: "El contrato entra en vigor mañana", target: "The contract comes into effect tomorrow", status: "green" },
  { id: 2, source: "El vehículo debe inspeccionarse antes de la entrega", target: "The car must be inspected before delivery", status: "yellow", issues: ["Terminology inconsistency: 'vehicle' translated as 'car'"], aifix: "The vehicle must be inspected before delivery" },
  { id: 3, source: "El seguro cubre todos los daños materiales y personales sin límite", target: "Insurance covers most material damages", status: "red", issues: ["Missing: 'personal'", "Missing: 'without limit'"], aifix: "The insurance covers all material and personal damages without limit" },
  { id: 4, source: "Las disputas se resolverán en los tribunales de Madrid", target: "Disputes shall be resolved in the courts of Madrid", status: "green" }
];

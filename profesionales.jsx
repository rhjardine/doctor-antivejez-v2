import { useState } from "react";
import { Button, Input, Table, Switch } from "@/components/ui";
import { Trash, Eye, Edit } from "lucide-react";

const professionalsData = [
  { id: 1, name: "Richard Jardine", idCard: "V-12431453", email: "rhjardine@gmail.com", coach: "No", status: "Activo", forms: 0 },
  { id: 2, name: "Carolina Salvatori", idCard: "E-23592103", email: "dracarolinasalvatori@gmail.com", coach: "No", status: "Activo", forms: 50 },
  { id: 3, name: "Silvina Plaza", idCard: "E-27388831", email: "silvina_plaza@hotmail.com", coach: "No", status: "Activo", forms: 50 },
];

export default function Professionals() {
  const [professionals, setProfessionals] = useState(professionalsData);
  const [search, setSearch] = useState("");

  const handleDelete = (id) => {
    setProfessionals(professionals.filter((prof) => prof.id !== id));
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-2xl font-bold text-gray-800">Profesionales</h1>
      <div className="flex justify-between my-4">
        <Input
          placeholder="Buscar por nombre, cédula o correo"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-1/3"
        />
        <div className="flex gap-2">
          <Button>Crear Profesional</Button>
          <Button variant="outline">Exportar Excel</Button>
        </div>
      </div>
      <Table>
        <thead>
          <tr>
            <th>Nombre y Apellido</th>
            <th>Cédula</th>
            <th>Correo</th>
            <th>Coach</th>
            <th>Estado</th>
            <th>Formularios</th>
            <th>Acciones</th>
          </tr>
        </thead>
        <tbody>
          {professionals.map((prof) => (
            <tr key={prof.id} className="bg-white">
              <td>{prof.name}</td>
              <td>{prof.idCard}</td>
              <td>{prof.email}</td>
              <td>{prof.coach}</td>
              <td>{prof.status}</td>
              <td>{prof.forms}</td>
              <td className="flex gap-2">
                <Switch />
                <Button variant="ghost" size="icon">
                  <Eye className="w-4 h-4" />
                </Button>
                <Button variant="ghost" size="icon" onClick={() => handleDelete(prof.id)}>
                  <Trash className="w-4 h-4 text-red-500" />
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
}
